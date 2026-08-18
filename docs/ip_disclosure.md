# Engineering Invention Disclosure & Prior Art Analysis Package

## 1. Technical Field
Vector similarity retrieval, multidimensional geometric index structures, complex-valued projective embeddings, and admissible Cauchy-Schwarz hierarchical bounding over normalized Riemannian hyperspheres ($S^{D-1}$).

---

## 2. Problem Statement
Traditional Approximate Nearest Neighbor (ANN) indexes (such as HNSW, IVF-PQ, and ScaNN) are fundamentally heuristic:
1. They lack admissible certificates of exactness, suffering catastrophic recall collapse in high-dimensional or out-of-distribution regimes.
2. Exact retrieval approaches either incur exhaustive $O(ND)$ SIMD compute or employ loose Euclidean Cauchy-Schwarz bounds that fail to prune the hypersphere effectively.
3. Quantized filtering (LUTz / FastScan) when applied globally across the entire corpus degrades into scanning overhead.

---

## 3. The 4-Stage Hierarchical Proof Invariant
HNSQR decomposes exact high-dimensional retrieval into four decoupled, provable stages:

$$\text{Rivero proposes} \implies \text{ProofTree eliminates} \implies \text{LUTz prunes} \implies \text{SIMD decides}$$

1. **Rivero Bounded Proposal Generation**:
   - Compiles high-dimensional complex vectors via complex projective quantization ($\mathbb{C}^D \to \text{SimHash}$).
   - Bounds proposal generation to an $O(1)$ hardware-friendly work ceiling independent of corpus cardinality $N$.
2. **Spherical-Cap Admissible Bounding ($\text{UB}_{\text{cap}}$)**:
   - For territory $T \subset S^{D-1}$ with normalized centroid $\hat{c}_T$ and angular radius $\theta_T = \max_{x \in T} \arccos(\hat{c}_T^\top x)$:
     $$\text{UB}_{\text{cap}}(q, T) = \begin{cases} 1.0 & \text{if } (q^\top \hat{c}_T) \ge \cos\theta_T \\ (q^\top \hat{c}_T) \cos\theta_T + \sqrt{1 - (q^\top \hat{c}_T)^2} \sin\theta_T & \text{otherwise} \end{cases}$$
   - Eliminates **$84\% - 93\%$** of the corpus without evaluating a single vector coordinate.
3. **Progressive LUTz Leaf Cascade**:
   - Evaluates 4-bit block-quantized L0 and L1 Cauchy-Schwarz upper bounds exclusively on unresolved proof leaves with leaf-local winner-first scheduling.
   - Eliminates all but **$0.90\% - 1.89\%$** of candidates.
4. **Exact SIMD Resolution**:
   - Evaluates full float precision exclusively on genuine residue.
   - Guaranteed $100.0000\%$ exact Top-K equality with deterministic lexicographic tie-breaking.

---

## 4. Prior Art Distinction

| System / Technique | Metric Space | Exact Guarantee | Pruning Geometry | Work Scaling |
| :--- | :--- | :--- | :--- | :--- |
| **HNSW** | Arbitrary | Heuristic ($\approx 90-98\%$) | Proximity Graph | $O(\log N)$ heuristic |
| **IVF-PQ** | Euclidean | Heuristic ($\approx 80-95\%$) | Voronoi cells | $O(N_{\text{probed}})$ |
| **ScaNN** | Cosine / MIPS | Heuristic | Anisotropic Quantization | $O(N_{\text{probed}})$ |
| **HNSQR (Certified)** | Normalized Complex Sphere | **100.0000% Provable** | **Spherical-Cap + Block Cauchy-Schwarz** | **$< 1\% - 2\%$ SIMD Residue** |

---

## 5. Development Chronology & Evidence Record
- **Gate A (Fixed Address & Dimensional Aliasing Diagnostics)**: Established that coarse routing width must scale with dimension.
- **Gate B0/B1 (Corpus-Covering Proof Substrate)**: Implemented flattened tree with disjoint complete partition and $f64$ Cauchy-Schwarz bounds.
- **Gate B2 (Spherical-Cap Geometry & Angular 2-Means)**: Discovered that Euclidean balls permit collinear residual leakage; spherical caps increased pruning from $< 1\%$ to $> 86\%$.
- **Gate B3 (LUTz L0/L1 Leaf Cascade & Normalized Accounting)**: Reduced exact SIMD evaluations to $0.90\% - 1.89\%$ with $100.00\%$ normalized terminal accounting.
