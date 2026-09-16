use std::cmp::Ordering;

pub const EPS: f64 = 1.0e-12;

#[inline]
pub fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

#[inline]
pub fn logit(p: f64) -> f64 {
    let p = clip(p, 1.0e-4, 1.0 - 1.0e-4);
    (p / (1.0 - p)).ln()
}

#[inline]
pub fn sigmoid(z: f64) -> f64 {
    let z = clip(z, -25.0, 25.0);
    1.0 / (1.0 + (-z).exp())
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn rms(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().max(1) as f64;
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n).sqrt()
}

pub fn canonicalize_direction(mut u: Vec<f64>) -> Vec<f64> {
    let n = norm(&u);
    if n <= EPS { return u; }
    for x in &mut u { *x /= n; }
    if let Some((j, _)) = u.iter().enumerate().max_by(|(_, a), (_, b)| {
        a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal)
    }) {
        if u[j] < 0.0 {
            for x in &mut u { *x = -*x; }
        }
    }
    u
}

/// Symmetric Jacobi eigensolver for small dense matrices.
/// Returns (eigenvalues, eigenvectors), where eigenvector `j` is the j-th column.
pub fn jacobi_eigen_symmetric(a: &[f64], n: usize, max_sweeps: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n);
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n { v[i * n + i] = 1.0; }

    for _ in 0..max_sweeps {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let x = m[i * n + j].abs();
                if x > max_off { max_off = x; p = i; q = j; }
            }
        }
        if max_off <= tol { break; }

        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];
        if apq.abs() <= tol { continue; }

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..n {
            if k != p && k != q {
                let mkp = m[k * n + p];
                let mkq = m[k * n + q];
                let npv = c * mkp - s * mkq;
                let nqv = s * mkp + c * mkq;
                m[k * n + p] = npv; m[p * n + k] = npv;
                m[k * n + q] = nqv; m[q * n + k] = nqv;
            }
        }
        m[p * n + p] = app - t * apq;
        m[q * n + q] = aqq + t * apq;
        m[p * n + q] = 0.0;
        m[q * n + p] = 0.0;

        for k in 0..n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }

    let vals = (0..n).map(|i| m[i * n + i]).collect();
    (vals, v)
}

pub fn column(m: &[f64], nrows: usize, ncols: usize, j: usize) -> Vec<f64> {
    debug_assert_eq!(m.len(), nrows * ncols);
    (0..nrows).map(|i| m[i * ncols + j]).collect()
}
