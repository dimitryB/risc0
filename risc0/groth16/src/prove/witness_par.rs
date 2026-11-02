// Parallel evaluation of the shrink_wrap witness graph.
//
// The graph (18.1M nodes, depth 87,607) is cut into bands of BAND consecutive ASAP levels.
// Inside a band, nodes are grouped into connected components, ignoring operands from earlier
// bands and leaves (inputs/constants), whose values are final before the band starts. Every
// component is evaluated by one thread in node (topological) order, once every component it
// reads from is done (dataflow over the component DAG). Each node is written exactly once, by
// the task that owns its component, and only reads operands that are (a) leaves, (b) in an
// earlier band, or (c) earlier in its own component: no data races, and the values are the same
// as the serial evaluator's.
//
// Values live in a node-indexed array (18.1M x 32 B = 580 MB), allocated and pre-faulted once
// at compile time (on the graph preload thread) and reused by every proof.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Mutex,
};

use circom_witnesscalc::{
    field::{FieldOperations, U254},
    graph::{Node, Nodes, NodesStorage, VecNodes},
};

use super::witness_fast::{
    add_mod, compile_node_ins, from_l, mont_mul, operands, sub_mod, to_l, Ins, NodeIns, L,
};

pub(crate) struct ParProgram {
    ni: NodeIns,
    leaves: Vec<u32>,
    /// Component node lists, flattened; comp_off[c]..comp_off[c+1].
    comp_nodes: Vec<u32>,
    comp_off: Vec<u32>,
    out_idx: Vec<u32>,
    out_dups: Vec<(u32, u32)>,
    /// Dataflow schedule: component DAG (edges from operand components to users).
    succ: Vec<u32>,
    succ_off: Vec<u32>,
    indeg: Vec<u32>,
    roots: Vec<u32>,
    cnt: Vec<AtomicU32>,
    values: Mutex<Vec<L>>,
    pool: rayon::ThreadPool,
}

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        let g = p[p[x as usize] as usize];
        p[x as usize] = g;
        x = g;
    }
    x
}

/// ASAP levels per band.
const BAND: u32 = 32;
/// Evaluation threads (measured best on a 16-core host; more threads contend for memory).
const THREADS: usize = 12;

pub(crate) fn compile(nodes: &Nodes<U254, VecNodes>, signals: &[usize]) -> Option<ParProgram> {
    let band = BAND;
    let nthr = THREADS;
    let ni = compile_node_ins(nodes, signals)?;
    let n = nodes.nodes.len();
    let mut level = vec![0u32; n];
    let mut leaf = vec![false; n];
    for i in 0..n {
        let ops = operands(&nodes.nodes.get(i).unwrap());
        if ops.iter().all(|&d| d == usize::MAX) {
            leaf[i] = true;
            continue;
        }
        level[i] = ops
            .iter()
            .filter(|&&d| d != usize::MAX)
            .map(|&d| level[d] + 1)
            .max()
            .unwrap();
    }
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        if leaf[i] {
            continue;
        }
        for d in operands(&nodes.nodes.get(i).unwrap()) {
            if d != usize::MAX && !leaf[d] && level[d] / band == level[i] / band {
                let (x, y) = (find(&mut p, i as u32), find(&mut p, d as u32));
                if x != y {
                    p[x as usize] = y;
                }
            }
        }
    }
    let mut comp_id = vec![u32::MAX; n]; // by root
    let mut comp_size: Vec<u32> = Vec::new();
    let mut comp_band: Vec<u32> = Vec::new();
    let mut node_comp = vec![u32::MAX; n];
    let mut leaves = Vec::new();
    for i in 0..n {
        if leaf[i] {
            leaves.push(i as u32);
            continue;
        }
        let r = find(&mut p, i as u32) as usize;
        if comp_id[r] == u32::MAX {
            comp_id[r] = comp_size.len() as u32;
            comp_size.push(0);
            comp_band.push(level[i] / band);
        }
        let c = comp_id[r];
        node_comp[i] = c;
        comp_size[c as usize] += 1;
    }
    let nc = comp_size.len();
    let mut comp_off = vec![0u32; nc + 1];
    for c in 0..nc {
        comp_off[c + 1] = comp_off[c] + comp_size[c];
    }
    let mut fill = comp_off.clone();
    let mut comp_nodes = vec![0u32; comp_off[nc] as usize];
    for (i, &c) in node_comp.iter().enumerate().take(n) {
        if c != u32::MAX {
            comp_nodes[fill[c as usize] as usize] = i as u32;
            fill[c as usize] += 1;
        }
    }
    // Component DAG: an edge for every operand in another (earlier-band) component.
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        let c = node_comp[i];
        if c == u32::MAX {
            continue;
        }
        for d in operands(&nodes.nodes.get(i).unwrap()) {
            if d != usize::MAX {
                let cd = node_comp[d];
                if cd != u32::MAX && cd != c {
                    edges.push((cd, c));
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let mut succ_off = vec![0u32; nc + 1];
    let mut indeg = vec![0u32; nc];
    for &(s, d) in &edges {
        succ_off[s as usize + 1] += 1;
        indeg[d as usize] += 1;
    }
    for c in 0..nc {
        succ_off[c + 1] += succ_off[c];
    }
    let mut succ: Vec<u32> = edges.iter().map(|e| e.1).collect();
    drop(edges);
    for c in 0..nc {
        let (s, e) = (succ_off[c] as usize, succ_off[c + 1] as usize);
        succ[s..e].sort_by(|&x, &y| comp_size[y as usize].cmp(&comp_size[x as usize]));
    }
    let mut roots: Vec<u32> = (0..nc as u32).filter(|&c| indeg[c as usize] == 0).collect();
    roots.sort_by(|&x, &y| comp_size[y as usize].cmp(&comp_size[x as usize]));
    let cnt: Vec<AtomicU32> = indeg.iter().map(|&d| AtomicU32::new(d)).collect();
    let mut out_idx = vec![u32::MAX; n];
    let mut out_dups = Vec::new();
    for (w, &s) in signals.iter().enumerate() {
        if out_idx[s] == u32::MAX {
            out_idx[s] = w as u32;
        } else {
            out_dups.push((w as u32, out_idx[s]));
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthr)
        .thread_name(|i| format!("g16-witness-{i}"))
        .build()
        .ok()?;
    // Fault the value array in now (off the proving path): vec![0; n] is calloc'd, and the
    // first evaluation would otherwise take ~150-200 ms of page faults.
    let mut values = vec![[0u64; 4]; n];
    for chunk in values.chunks_mut(128) {
        unsafe { std::ptr::write_volatile(&mut chunk[0], [1, 0, 0, 0]) };
    }
    Some(ParProgram {
        ni,
        leaves,
        comp_nodes,
        comp_off,
        out_idx,
        out_dups,
        succ,
        succ_off,
        indeg,
        roots,
        cnt,
        values: Mutex::new(values),
        pool,
    })
}

struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

#[inline(always)]
unsafe fn eval_node(
    i: usize,
    ni: &NodeIns,
    nodes: &Nodes<U254, VecNodes>,
    inputs: &[U254],
    v: *const L,
) -> L {
    let g = |j: u32| &*v.add(j as usize);
    match *ni.ins.get_unchecked(i) {
        Ins::MulVV(a, b) => mont_mul(&mont_mul(g(a), g(b)), &ni.r2),
        Ins::MulVK(a, k) => mont_mul(g(a), ni.kmul.get_unchecked(k as usize)),
        Ins::Add(a, b) => add_mod(g(a), g(b)),
        Ins::AddVK(a, k) => add_mod(g(a), ni.kraw.get_unchecked(k as usize)),
        Ins::Sub(a, b) => sub_mod(g(a), g(b)),
        Ins::And(a, k) => {
            let (x, c) = (g(a), ni.kraw.get_unchecked(k as usize));
            [x[0] & c[0], x[1] & c[1], x[2] & c[2], x[3] & c[3]]
        }
        Ins::Slow => {
            let ff = &nodes.ff;
            let gv = |j: usize| from_l(g(j as u32));
            let val: U254 = match nodes.nodes.get(i).unwrap() {
                Node::Unknown => panic!("Unknown node"),
                Node::Constant(c) => nodes.constants[c],
                Node::Input(j) => inputs[j],
                Node::Op(op, a, b) => ff.op_duo(op, gv(a), gv(b)),
                Node::UnoOp(op, a) => ff.op_uno(op, gv(a)),
                Node::TresOp(_, a, b, c) => {
                    if gv(a) == U254::ZERO {
                        gv(c)
                    } else {
                        gv(b)
                    }
                }
            };
            to_l(&val)
        }
    }
}

/// Evaluate the witness into `out` (one canonical little-endian value per signal).
pub(crate) fn eval(p: &ParProgram, nodes: &Nodes<U254, VecNodes>, inputs: &[U254], out: &mut [L]) {
    assert_eq!(p.ni.ins.len(), nodes.nodes.len());
    assert!(p
        .out_idx
        .iter()
        .all(|&w| w == u32::MAX || (w as usize) < out.len()));
    assert!(p
        .out_dups
        .iter()
        .all(|&(w, s)| (w as usize) < out.len() && (s as usize) < out.len()));
    let mut guard = p.values.lock().unwrap();
    let v = SyncPtr(guard.as_mut_ptr());
    let o = SyncPtr(out.as_mut_ptr());
    let run = |i: u32, v: &SyncPtr<L>, o: &SyncPtr<L>| unsafe {
        let x = eval_node(i as usize, &p.ni, nodes, inputs, v.0);
        *v.0.add(i as usize) = x;
        let w = *p.out_idx.get_unchecked(i as usize);
        if w != u32::MAX {
            *o.0.add(w as usize) = x;
        }
    };
    for &i in &p.leaves {
        run(i, &v, &o);
    }
    let run_comp = |c: u32| {
        let (s, e) = (p.comp_off[c as usize], p.comp_off[c as usize + 1]);
        for &i in &p.comp_nodes[s as usize..e as usize] {
            run(i, &v, &o);
        }
    };
    {
        // Dataflow: a component runs once all components it reads from are done (the AcqRel
        // decrement that releases it orders their writes before its reads). No barriers, so
        // one descheduled thread delays only its own successors instead of every band.
        for (c, &d) in p.cnt.iter().zip(p.indeg.iter()) {
            c.store(d, Ordering::Relaxed);
        }
        fn go<'a, F: Fn(u32) + Sync>(
            mut c: u32,
            sc: &rayon::Scope<'a>,
            p: &'a ParProgram,
            f: &'a F,
        ) {
            loop {
                f(c);
                let mut next = u32::MAX;
                for k in p.succ_off[c as usize]..p.succ_off[c as usize + 1] {
                    let d = p.succ[k as usize];
                    if p.cnt[d as usize].fetch_sub(1, Ordering::AcqRel) == 1 {
                        if next == u32::MAX {
                            next = d;
                        } else {
                            sc.spawn(move |sc| go(d, sc, p, f));
                        }
                    }
                }
                if next == u32::MAX {
                    break;
                }
                c = next;
            }
        }
        let f = &run_comp;
        p.pool.install(|| {
            rayon::scope(|sc| {
                for &r in &p.roots {
                    sc.spawn(move |sc| go(r, sc, p, f));
                }
            })
        });
        debug_assert!(p.cnt.iter().all(|c| c.load(Ordering::Relaxed) == 0));
    }
    drop(guard);
    for &(w, src) in &p.out_dups {
        out[w as usize] = out[src as usize];
    }
}
