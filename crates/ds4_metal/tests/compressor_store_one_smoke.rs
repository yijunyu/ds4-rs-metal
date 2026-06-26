//! M5 Phase D — `kernel_dsv4_compressor_store_one` MSL port smoke.
//!
//! Compares our inline-MSL port (verbatim from antirez `dsv4_kv.metal:201`)
//! against a hand-rolled CPU oracle of the same algorithm. The kernel is
//! bit-deterministic (no reductions, no transcendentals — just an APE
//! lookup + one float add + two writes), so the CPU vs GPU outputs must
//! be bit-identical.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use ds4_metal::MetalDispatcher;

/// Mirror of the MSL kernel body. Writes `state_kv[dst] = kv[i]` and
/// `state_score[dst] = score[i] + ape[pos_mod * width + i]` for each
/// element `i in 0..width`. `dst_row = ratio + (pos%ratio)` when
/// `ratio == 4`, else `pos % ratio`. Initial state buffers are zero.
fn cpu_oracle(
    state_kv: &mut [f32],
    state_score: &mut [f32],
    kv: &[f32],
    score: &[f32],
    ape: &[f32],
    width: usize,
    ratio: usize,
    pos: usize,
) {
    let pos_mod = pos % ratio;
    let dst_row = if ratio == 4 { ratio + pos_mod } else { pos_mod };
    for i in 0..width {
        let dst = dst_row * width + i;
        let ape_i = pos_mod * width + i;
        state_kv[dst] = kv[i];
        state_score[dst] = score[i] + ape[ape_i];
    }
}

fn read_state(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents() as *const f32,
            out.as_mut_ptr(),
            n,
        );
    }
    out
}

fn run_case(
    disp: &MetalDispatcher,
    width: u32,
    ratio: u32,
    pos: u32,
    seed: u32,
) {
    let w = width as usize;
    let r = ratio as usize;
    let state_rows = if r == 4 { 2 * r } else { r };
    let state_len = state_rows * w;

    let kv: Vec<f32> = (0..w)
        .map(|i| ((i as f32 + seed as f32) * 0.013).sin() * 0.4)
        .collect();
    let score: Vec<f32> = (0..w)
        .map(|i| ((i as f32 + seed as f32) * 0.017).cos() * 0.3)
        .collect();
    let ape: Vec<f32> = (0..r * w)
        .map(|i| (i as f32 * 0.011 - 0.2).sin() * 0.25)
        .collect();
    // CPU oracle.
    let mut ref_state_kv = vec![0.0f32; state_len];
    let mut ref_state_score = vec![0.0f32; state_len];
    cpu_oracle(
        &mut ref_state_kv,
        &mut ref_state_score,
        &kv,
        &score,
        &ape,
        w,
        r,
        pos as usize,
    );

    // GPU path: zero-init state buffers via the scope's allocator,
    // dispatch the kernel, drain the cb, read state buffers.
    let mut scope = disp.batch_scope();
    let state_kv_db = scope.alloc_f32(state_len);
    let state_score_db = scope.alloc_f32(state_len);
    let ape_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            ape.as_ptr() as *const u8,
            ape.len() * std::mem::size_of::<f32>(),
        )
    };
    scope
        .compressor_store_one(
            &kv,
            &score,
            ape_bytes,
            state_kv_db.buffer(),
            state_score_db.buffer(),
            width,
            ratio,
            pos,
            false, // ape is f32 here
        )
        .expect("compressor_store_one");
    // No DeferredBuf outputs — drain the cb so writes are CPU-visible.
    let _ = scope.commit_wait_read_multi(&[]);

    let gpu_state_kv = read_state(state_kv_db.buffer(), state_len);
    let gpu_state_score = read_state(state_score_db.buffer(), state_len);

    for i in 0..state_len {
        assert!(
            (gpu_state_kv[i] - ref_state_kv[i]).abs() < 1e-7,
            "state_kv[{i}] differs: gpu={} ref={} (width={width} ratio={ratio} pos={pos} seed={seed})",
            gpu_state_kv[i],
            ref_state_kv[i],
        );
        assert!(
            (gpu_state_score[i] - ref_state_score[i]).abs() < 1e-6,
            "state_score[{i}] differs: gpu={} ref={} (width={width} ratio={ratio} pos={pos} seed={seed})",
            gpu_state_score[i],
            ref_state_score[i],
        );
    }
}

#[test]
fn compressor_store_one_matches_cpu_oracle_ratio4() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // V4 Flash uses ratio=4, head_dim=128 for compressor; smaller width
    // here just exercises the kernel arithmetic. Multiple pos values
    // span pos_mod = 0..ratio so every dst_row is visited.
    for pos in 0..8 {
        run_case(&disp, 128, 4, pos, pos);
    }
}

#[test]
fn compressor_store_one_matches_cpu_oracle_ratio1() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // ratio==1 edge case: dst_row == pos_mod == 0 always; state buffer
    // is single-row.
    for pos in 0..4 {
        run_case(&disp, 64, 1, pos, pos);
    }
}

#[test]
fn compressor_store_one_matches_cpu_oracle_small_width() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");
    // width below the threadgroup width (32) — exercises the
    // bounds check in the kernel.
    run_case(&disp, 17, 4, 3, 99);
}

/// Persistent per-layer state pool: repeated `compressor_state_*_or_alloc`
/// calls on the same layer return the same buffer; writes from prior
/// dispatches accumulate; different layers and different kinds
/// (kv vs score, compressor vs indexer) don't collide.
#[test]
fn compressor_state_pool_persists_across_calls() {
    let disp = MetalDispatcher::new().expect("MetalDispatcher::new");

    const WIDTH: u32 = 128;
    const RATIO: u32 = 4;
    let state_rows = 2 * RATIO as usize; // ratio==4 → 2*ratio rows
    let state_len = state_rows * WIDTH as usize;
    let state_bytes = state_len * std::mem::size_of::<f32>();

    // Allocate per-layer state for layer 5 (arbitrary).
    let layer_idx = 5u32;
    let kv_buf_a = disp.compressor_state_kv_or_alloc(layer_idx, state_bytes);
    let score_buf_a = disp.compressor_state_score_or_alloc(layer_idx, state_bytes);

    // Re-alloc returns the same buffer (Arc identity preserved).
    let kv_buf_b = disp.compressor_state_kv_or_alloc(layer_idx, state_bytes);
    let score_buf_b = disp.compressor_state_score_or_alloc(layer_idx, state_bytes);
    assert!(
        std::ptr::eq(
            kv_buf_a.contents() as *const u8,
            kv_buf_b.contents() as *const u8,
        ),
        "compressor_state_kv pool didn't return identical buffer on re-alloc"
    );
    assert!(
        std::ptr::eq(
            score_buf_a.contents() as *const u8,
            score_buf_b.contents() as *const u8,
        ),
        "compressor_state_score pool didn't return identical buffer on re-alloc"
    );

    // Different layer → different buffer.
    let other_layer = 6u32;
    let kv_other = disp.compressor_state_kv_or_alloc(other_layer, state_bytes);
    assert!(
        !std::ptr::eq(
            kv_buf_a.contents() as *const u8,
            kv_other.contents() as *const u8,
        ),
        "different layers shared the same buffer"
    );

    // Different kind → different buffer (even on the same layer).
    assert!(
        !std::ptr::eq(
            kv_buf_a.contents() as *const u8,
            score_buf_a.contents() as *const u8,
        ),
        "kv and score pools collided"
    );
    let indexer_kv = disp.indexer_state_kv_or_alloc(layer_idx, state_bytes);
    assert!(
        !std::ptr::eq(
            kv_buf_a.contents() as *const u8,
            indexer_kv.contents() as *const u8,
        ),
        "compressor_kv and indexer_kv pools collided"
    );

    // Accumulating writes across pos=0..ratio: each dispatch overwrites
    // exactly one row of the state buffer (dst_row = ratio + pos%ratio
    // for ratio==4). After running pos=0..7, every row in [ratio,
    // 2*ratio) is written; the [0, ratio) prefix stays at its initial
    // zero (we don't write that region for ratio==4).
    let ape: Vec<f32> = (0..(RATIO as usize) * (WIDTH as usize))
        .map(|i| (i as f32 * 0.011 - 0.2).sin() * 0.25)
        .collect();
    let ape_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            ape.as_ptr() as *const u8,
            ape.len() * std::mem::size_of::<f32>(),
        )
    };
    // Use a fresh layer to keep this test independent of the
    // ordering of prior allocs in this same MetalDispatcher.
    let li = 42u32;
    let kv_st = disp.compressor_state_kv_or_alloc(li, state_bytes);
    let sc_st = disp.compressor_state_score_or_alloc(li, state_bytes);

    // Build per-pos input rows (deterministic but distinct).
    let mk_kv = |pos: u32| -> Vec<f32> {
        (0..WIDTH as usize)
            .map(|i| ((i as f32 + pos as f32 * 7.0) * 0.013).sin() * 0.4)
            .collect()
    };
    let mk_score = |pos: u32| -> Vec<f32> {
        (0..WIDTH as usize)
            .map(|i| ((i as f32 + pos as f32 * 11.0) * 0.017).cos() * 0.3)
            .collect()
    };

    for pos in 0..(2 * RATIO) {
        let kv = mk_kv(pos);
        let score = mk_score(pos);
        let mut scope = disp.batch_scope();
        scope
            .compressor_store_one(
                &kv, &score, ape_bytes, &kv_st, &sc_st, WIDTH, RATIO, pos, false,
            )
            .expect("compressor_store_one");
        let _ = scope.commit_wait_read_multi(&[]);
    }

    // CPU oracle: replay the same sequence into Vec<f32>s. Initial
    // values match the persistent GPU state pools: state_kv is
    // zero-init, state_score is filled with -1e9 (softmax-mask
    // sentinel matching `decode_step.rs:493`).
    let mut ref_kv = vec![0.0f32; state_len];
    let mut ref_score = vec![-1.0e9f32; state_len];
    for pos in 0..(2 * RATIO) {
        let kv = mk_kv(pos);
        let score = mk_score(pos);
        cpu_oracle(
            &mut ref_kv,
            &mut ref_score,
            &kv,
            &score,
            &ape,
            WIDTH as usize,
            RATIO as usize,
            pos as usize,
        );
    }

    let gpu_kv = read_state(&kv_st, state_len);
    let gpu_score = read_state(&sc_st, state_len);
    for i in 0..state_len {
        assert!(
            (gpu_kv[i] - ref_kv[i]).abs() < 1e-7,
            "accumulated state_kv[{i}] differs: gpu={} ref={}",
            gpu_kv[i],
            ref_kv[i],
        );
        assert!(
            (gpu_score[i] - ref_score[i]).abs() < 1e-6,
            "accumulated state_score[{i}] differs: gpu={} ref={}",
            gpu_score[i],
            ref_score[i],
        );
    }

    // The first ratio rows of the state buffer (rows 0..ratio for
    // ratio==4) are never written by compressor_store_one — they're
    // the "prev window" half. Sanity: kv stays at its zero init,
    // score stays at its -1e9 softmax-mask init (so the pool kernel
    // treats them as masked-out when it runs at pos=ratio-1).
    for i in 0..(RATIO as usize) * (WIDTH as usize) {
        assert_eq!(
            gpu_kv[i], 0.0,
            "ratio==4 'prev window' state_kv at {i} should be zero",
        );
        assert_eq!(
            gpu_score[i], -1.0e9,
            "ratio==4 'prev window' state_score at {i} should be -1e9 sentinel",
        );
    }
}
