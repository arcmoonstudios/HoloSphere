2026-03-30 - CPQ-8 Asymmetric Inner Product Unrolling and Scaling Precomputation
Learning: Precomputing the scalar amplitude scale step `step_r = (max_r - min_r) / 255.0` outside the hot loop in `asymmetric_dot_product_raw` removes a floating point multiplication per complex component. 4-way loop unrolling with `chunks_exact(4)` / `chunks_exact(8)` eliminates bounds checks and enables vector instruction pipelining.
Action: Prefer precomputing loop invariants in vector distance inner loops and use `chunks_exact` to hint bound check removal to LLVM.
