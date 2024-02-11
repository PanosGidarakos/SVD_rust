extern crate ndarray;
use ndarray::{Array1, s, Array2,ArrayBase, Data, Ix2,};
use std::f64;

fn apply_givens_lefts_on_right(u: &mut Array2<f64>, lefts: Vec<(f64, f64, usize, usize)>, shift_dex: usize) {
    for curr_left in lefts {
        let (c, s, mut c1, mut c2) = curr_left;
        c1 += shift_dex; 
        c2 += shift_dex;
        right_givens(u, (c, s, c1, c2));
    }
}
fn apply_givens_rights_on_left(rights: Vec<(f64, f64, usize, usize)>, vt: &mut Array2<f64>, shift_dex: usize) {
    for curr_right in rights {
        let (c, s, mut r1, mut r2) = curr_right;
        r1 += shift_dex; 
        r2 += shift_dex;
        left_givens_transposed(vt, (c, s, r1, r2)); // Assuming left_givensT is similar to this
    }
}
fn find_first_run_boolean(vec: &[bool], front: bool) -> std::ops::Range<usize> {
    let mut start = None;
    let mut end = 0;

    for (idx, &itm) in vec.iter().enumerate() {
        let actual_idx = if front { idx } else { vec.len() - 1 - idx };

        if itm {
            if start.is_none() {
                start = Some(actual_idx);
            }
            end = actual_idx;
        } else if start.is_some() {
            break;
        }
    }

    if let Some(start_idx) = start {
        // Adjust the end of the range to be non-inclusive by adding 1
        // Ensure this doesn't overflow, though in this context it's unlikely to be an issue
        start_idx..end + 1
    } else {
        0..0 // Equivalent to Range<usize>::empty()
    }
}
fn extract_packed_house_bidiag(h: &Array2<f64>, betas_u: Vec<f64>, betas_v: Vec<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let u = extract_house_reflection(h.clone(), &betas_u, "lower");
    let vt = extract_house_reflection(h.clone(), &betas_v, "upper");
    let b = extract_upper_bidiag(h);

    (u, b, vt)
}
fn extract_upper_bidiag<S>(m: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64>,
{
    let rows = m.nrows();
    let cols = m.ncols();
    // Ensure the output matrix B is of the same size as M
    let mut b = Array2::<f64>::zeros((rows, cols));

    for i in 0..cols {
        // Set diagonal element
        b[[i, i]] = m[[i, i]];
        // Check and set superdiagonal element if within bounds
        if i + 1 < cols {
            b[[i, i + 1]] = m[[i, i + 1]];
        }
    }

    b
}
fn remove_small(b: &mut Array2<f64>, eps: f64) {
    let (m, n) = b.dim();
    // Diagonal elements
    for i in 0..m.min(n) {
        let val = b[(i, i)].abs();
        b[(i, i)] = if val < eps { 0.0 } else { val };
    }
    // Superdiagonal elements
    for i in 0..m.min(n - 1) {
        let diag_val = b[(i, i)].abs();
        let next_diag_val = b[(i + 1, i + 1)].abs();
        let small_val = eps * (diag_val + next_diag_val);
        let super_diag_val = b[(i, i + 1)].abs();
        b[(i, i + 1)] = if super_diag_val < small_val { 0.0 } else { super_diag_val };
    }
}

fn clean_and_partition(b: &mut Array2<f64>, eps: f64) -> (Array2<f64>, usize) {
    remove_small(b, eps);
    // Convert superdiagonal elements to a boolean vector
    let mut super_diag_bools = Vec::new();
    for i in 0..b.nrows().min(b.ncols()) - 1 {
        let val = b[(i, i + 1)].abs();
        super_diag_bools.push(val > eps);
    }
    // Use find_first_run_boolean on the boolean vector
    let super_slice = find_first_run_boolean(&super_diag_bools, true);
    if super_slice.end == 0 {
        return (Array2::from_elem((0, 0), 0.0), 0);
    }
    // Adjust slicing to use the correct indices
    let full_slice = s![super_slice.start..super_slice.end + 1, super_slice.start..super_slice.end + 1];
    let sub_matrix = b.slice(full_slice).to_owned();
    (sub_matrix, super_slice.start)
}

fn make_house_vec(x: &Array1<f64>) -> (Array1<f64>, f64) {
    // let n = x.len();
    let dot_1on = x.slice(s![1..]).dot(&x.slice(s![1..]));
    // Create a copy of x
    let mut v = x.to_owned();
    // Set v[0] to 1.0
    v[0] = 1.0;
    // let mut beta: f64;
    let  beta: f64;
    if dot_1on < f64::EPSILON {
        beta = 0.0;
    } else {
        let norm_x = (x[0] * x[0] + dot_1on).sqrt();
        if x[0] <= 0.0 {
            v[0] = x[0] - norm_x;
        } else {
            v[0] = -dot_1on / (x[0] + norm_x);
        }
        beta = 2.0 * v[0] * v[0] / (dot_1on + v[0] * v[0]);
        v /= v[0];
    }
    (v, beta)
}

fn outer_product(v1: &Array1<f64>, v2: &Array1<f64>) -> Array2<f64> {
    let len1 = v1.len();
    let len2 = v2.len();
    let mut result = Array2::zeros((len1, len2));
    for i in 0..len1 {
        for j in 0..len2 {
            result[[i, j]] = v1[i] * v2[j];
        }
    }
    result
}
fn extract_house_reflection(mut a: Array2<f64>, betas: &[f64], side: &str) -> Array2<f64> {
    let (mut m, mut n) = a.dim();
    if side == "upper" {
        a = a.t().to_owned(); // Transpose `a` for upper triangular processing
        std::mem::swap(&mut m, &mut n); // Swap dimensions to match the transposed matrix
    }
    let mut q = Array2::<f64>::eye(n); // Use `n` here as we're working with transposed dimensions for "upper"
    let mut v = Array1::zeros(n);
    for j in (0..n).rev() {
        v.slice_mut(s![j..]).fill(0.0); // Reset v from j to end
        v[j] = 1.0;
        for i in j..n {
            v[i] = a[[i, j]]; // Directly access the correct element, no shift needed
        }
        let mut mini_q = Array2::<f64>::eye(n - j);
        for i in 0..(n - j-1) {
            for k in 0..(n - j) {
                mini_q[[i, k]] -= betas[j] * v[i + j] * v[k + j]; // Adjust for offset `j` in `v`
            }
        }
        // Apply the transformation to the corresponding submatrix of `q`
        let mut q_slice = q.slice_mut(s![j.., j..]);
        q_slice.assign(&mini_q.dot(&q_slice));
    }
    if side == "upper" {
        q = q.t().to_owned(); // If we processed the "upper" side, transpose `q` back
    }
    q
}

fn house_bidiag(a: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let (m, n) = a.dim();
    assert!(m >= n, "The matrix A must have at least as many rows as columns.");

    let mut betas_u = Array1::<f64>::zeros(n);
    let mut betas_v = Array1::<f64>::zeros(n - 1);

    for j in 0..n {
        let (u, beta_u) = make_house_vec(&a.slice_mut(s![j.., j]).to_owned());
        betas_u[j] = beta_u;
        let temp = Array2::<f64>::eye(m - j);
        let temp2=temp.clone()-beta_u*outer_product(&u,&u);
        let _mini_house_u = temp2;
        let temp3=_mini_house_u.dot(&a.slice_mut(s![j.., j..]));
        a.slice_mut(s![j.., j..]).assign(&temp3);



        a.slice_mut(s![j+1.., j]).assign(&u.slice(s![1..]));


        if j < n - 1 {
            let (v, beta_v) = make_house_vec(&a.slice_mut(s![j, j+1..]).t().to_owned());

            betas_v[j] = beta_v;

            let temp = Array2::<f64>::eye(n - j-1);

            let temp2=temp.clone()-betas_v[j]*outer_product(&v,&v);

            let _mini_house_u = temp2;

            // let temp3=mini_house_u.dot(&a.slice_mut(s![j, j+1..]));

            a.slice_mut(s![j, j+2..]).assign(&v.slice(s![1..]));

        }
    }

    (betas_u, betas_v)
}
// fn generate_random_array(rows: usize, cols: usize) -> Array2<f64> {
//     let mut rng = rand::thread_rng();
//     let data: Vec<f64> = (0..rows * cols)
//         .map(|_| rng.gen_range(0.0..0.5)) // Generate random values between 0 and 1
//         .collect();

//     Array2::from_shape_vec((rows, cols), data).unwrap()
// }

fn zeroing_givens_coeffs(x: f64, z: f64) -> (f64, f64) {
    if z.abs() < f64::EPSILON {
        return (1.0, 0.0);
    }
    let r = (x.powi(2) + z.powi(2)).sqrt();
    (x / r, -z / r)
}
fn right_givens(a: &mut Array2<f64>, params: (f64, f64, usize, usize)) {
    let (c, s, c1, c2) = params;
    let (rows, _) = a.dim();
    for i in 0..rows {
        let temp1 = a[[i, c1]] * c + a[[i, c2]] * s;
        let temp2 = -a[[i, c1]] * s + a[[i, c2]] * c;
        a[[i, c1]] = temp1;
        a[[i, c2]] = temp2;
    }
}
fn left_givens_transposed(a: &mut Array2<f64>, params: (f64, f64, usize, usize)) {
    let (c, s, r1, r2) = params;
    let (_, cols) = a.dim();
    
    for j in 0..cols {
        let temp1 = c * a[[r1, j]] - s * a[[r2, j]];
        let temp2 = s * a[[r1, j]] + c * a[[r2, j]];
        a[[r1, j]] = temp1;
        a[[r2, j]] = temp2;
    }
}
fn givens_zero_adjacent_onband_left(a: &mut Array2<f64>, row: usize, col: usize) -> (f64, f64, usize, usize) {
    let (c, s) = zeroing_givens_coeffs(a[[row - 1, col]], a[[row, col]]);
    // Apply left Givens rotation (transpose equivalent)
    for i in 0..a.ncols() {
        let temp1 = c * a[[row - 1, i]] + s * a[[row, i]];
        let temp2 = -s * a[[row - 1, i]] + c * a[[row, i]];
        a[[row - 1, i]] = temp1;
        a[[row, i]] = temp2;
    }
    // Return the calculated rotation parameters and indices
    (c, s, row - 1, col)
}
fn givens_zero_adjacent_onband_right(a: &mut Array2<f64>, row: usize, col: usize) -> (f64, f64, usize, usize) {
    let (c, s) = zeroing_givens_coeffs(a[[row, col - 1]], a[[row, col]]);
    right_givens(a, (c, s, col - 1, col));
    // Return the calculated rotation parameters and indices
    (c, s, col - 1, col)
}
fn bounce_blemish_down_diagonal(bbm: &mut Array2<f64>) -> (Vec<(f64, f64, usize, usize)>, Vec<(f64, f64, usize, usize)>) {
    let n = bbm.ncols();
    let mut left_rotations = Vec::new(); // Collect left Givens rotations
    let mut right_rotations = Vec::new(); // Collect right Givens rotations
    for k in 0..n-2 {
        // Assume givens_zero_adjacent_onband_left/right update rotations vectors
        let left_rotation = givens_zero_adjacent_onband_left(bbm, k + 1, k);
        let right_rotation = givens_zero_adjacent_onband_right(bbm, k, k + 2);
        left_rotations.push(left_rotation);
        right_rotations.push(right_rotation);
    }
    let last_left_rotation = givens_zero_adjacent_onband_left(bbm, n - 1, n - 2);
    left_rotations.push(last_left_rotation);
    // Return the collected rotations
    (left_rotations, right_rotations)
}
fn decouple_or_reduce(b: &mut Array2<f64>, zero_idxs: Vec<usize>) -> (Vec<(f64, f64, usize, usize)>, Vec<(f64, f64, usize, usize)>) {
    let n =  b.ncols();
    let mut lefts = Vec::new();
    let mut rights = Vec::new();
    for &kk in zero_idxs.iter() {
        if kk < n - 1 {
            let rotations = walk_blemish_out_right(b, kk, kk + 1);
            rights.extend(rotations);
        } else if kk == n - 1 {
            let rotations = walk_blemish_out_up(b, kk - 1, kk);
            lefts.extend(rotations);
        }
    }
    (lefts, rights)
}
fn walk_blemish_out_right(b: &mut Array2<f64>, row: usize, start_col: usize) -> Vec<(f64, f64, usize, usize)> {
    let n = b.ncols();
    let mut rotations = Vec::new();
    for col in start_col..n {
        let rotation = givens_zero_wdiag_left_on_bbm(b, row, col);
        rotations.push(rotation);
    }
    rotations
}
fn givens_zero_wdiag_left_on_bbm(bbm: &mut Array2<f64>, row: usize, col: usize) -> (f64, f64, usize, usize) {
    let (c, s) = zeroing_givens_coeffs(bbm[[col, col]], bbm[[row, col]]);
    let old_blemish = bbm[[row, col]];
    bbm[[row, col]] = 0.0;
    bbm[[col, col]] = bbm[[col, col]] * c - old_blemish * s;
    if col < bbm.ncols() - 1 {
        let temp = bbm[[row, col + 1]];
        bbm[[row, col + 1]] = s * temp;
        bbm[[col, col + 1]] *= c;
    }
    (c, s, row, col)
}
fn walk_blemish_out_up(b: &mut Array2<f64>, start_row: usize, col: usize) -> Vec<(f64, f64, usize, usize)> {
    let mut rotations = Vec::new();
    for row in (0..=start_row).rev() {
        let rotation = givens_zero_wdiag_right_on_bbm(b, row, col);
        rotations.push(rotation);
    }
    rotations
}
fn givens_zero_wdiag_right_on_bbm(bbm: &mut Array2<f64>, row: usize, col: usize) -> (f64, f64, usize, usize) {
    let (c, s) = zeroing_givens_coeffs(bbm[[row, row]], bbm[[row, col]]);
    let old_blemish = bbm[[row, col]];
    bbm[[row, col]] = 0.0;
    bbm[[row, row]] = bbm[[row, row]] * c - old_blemish * s;
    if row > 0 {
        let temp = bbm[[row - 1, col]];
        bbm[[row - 1, col]] = s * temp;
        bbm[[row - 1, row]] *= c;
    }
    (c, s, row, col)
}
fn step_full_bidiagonal_towards_diagonal(b: &mut Array2<f64>) -> (Vec<(f64, f64, usize, usize)>, Vec<(f64, f64, usize, usize)>) {
    // let rows = b.nrows();
    let cols = b.ncols();
    // Extract the last two columns of B
    let last_cols = b.slice(s![.., cols-2..]);
    // Compute T as the trailing 2x2 submatrix of B^T.dot(B)
    let t = last_cols.t().dot(&last_cols);
    // Compute the Wilkinson shift
    let d = (t[[0, 0]] - t[[1, 1]]) / 2.0;
    let shift = t[[1, 1]] - t[[1, 0]].powi(2) / (d + d.signum() * ((d.powi(2) + t[[1, 0]].powi(2)).sqrt()));
    // Special case: Compute Givens rotation to zero the right element
    let t_00 = b.t().row(0).mapv(|x| x.powi(2)).sum();
    let t_01 = b.t().row(0).dot(&b.column(1));
    let (c, s) = zeroing_givens_coeffs(t_00 - shift, t_01);
    // Apply the Givens rotation
    let local_right = (c, s, 0, 1);
    right_givens(b, local_right.clone());
    // Bounce blemish down diagonal and collect rotations
    let (lefts, mut rights) = bounce_blemish_down_diagonal(b);
    // Prepend the local right Givens rotation without copying (assuming rights is Vec)
    rights.insert(0, local_right);
    (lefts, rights)
}
fn clear_svd(a: &mut Array2<f64>, eps: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (_m, _n) = a.dim();
    let both_betas = house_bidiag(a); // Returns betas for U and Vt, needs to be adapted to Rust return type
    // Destructure returned tuple to get U, B, and V matrices
    let (mut u, mut b, v) = extract_packed_house_bidiag(a, both_betas.0.to_vec(), both_betas.1.to_vec());
    let mut vt = v.t().to_owned();
    // Clean and partition B to get B22 and k
    let (mut b22,mut k) = clean_and_partition(&mut b, eps);
    let mut iteration_count = 0; // For debugging: limit iterations to prevent infinite loops
    while b22.nrows() > 0 && iteration_count < 10{
        // Calculate zero indices. This is more complex in Rust and depends on how B22 is represented
        let zero_idxs: Vec<usize> = b22.diag().iter().enumerate()
            .filter_map(|(idx, &val)| if val.abs() < eps { Some(idx) } else { None })
            .collect();
        // Decide on action based on presence of zero indices
        let (lefts, rights) = if !zero_idxs.is_empty() {
            decouple_or_reduce(&mut b22, zero_idxs)
        } else {
            step_full_bidiagonal_towards_diagonal(&mut b22)
        };
        // Apply Givens rotations
        apply_givens_lefts_on_right(&mut u, lefts, k);
        apply_givens_rights_on_left(rights, &mut vt,k);
        // Clean and partition B again to update B22 and k
        (b22,k) = clean_and_partition(&mut b, eps);
        iteration_count += 1;
    }
    (u, b, vt)
}

//keeeeeep
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let root_area = BitMapBackend::new("execution_times.png", (640, 480)).into_drawing_area();
//     root_area.fill(&WHITE)?;

//     let mut chart = ChartBuilder::on(&root_area)
//         .caption("Execution Time vs. Array Size", ("sans-serif", 30))
//         .x_label_area_size(35)
//         .y_label_area_size(35)
//         .build_cartesian_2d(10..230, 0f32..9f32)?;

//     chart.configure_mesh().draw()?;
//     let mut durations: Vec<f32> = Vec::new();
//     let mut durationsna: Vec<f32> = Vec::new();
//     let mut rng = rand::thread_rng();
//     let eps = 0.001;
//     // Assuming the setup for rng, eps, tolerance, etc., is done here

//     for size in (10..=220).step_by(40) {
//         //custom_svd
//         let mut a = generate_random_array(size, size);
//         let start_time = Instant::now(); // Start timing
//         _=clear_svd(&mut a,eps);
//         let duration = start_time.elapsed(); // Calculate elapsed time
//         durations.push(duration.as_secs_f32()); // Store the duration

//     }
//     println!("{:?}", durations);
//     for size in (10..=220).step_by(40) {
//         let matrix = random_matrix(size, size, &mut rng);
//         let start_time_na = Instant::now(); // Start timing
//         let _ = matrix.svd(true, true);
//         let duration_na= start_time_na.elapsed(); // Calculate elapsed time
//         durationsna.push(duration_na.as_secs_f32());
//     }
//     println!("{:?}", durationsna);

//     // Ensure the plotting logic is correctly placed and executed after the loop
//     let sizes: Vec<i32> = (10..=220).step_by(40).collect();
//     // let duration_secs: Vec<f32> = durations.iter().map(|d| d.as_secs_f32()).collect(); // Convert durations to seconds
//     chart.draw_series(LineSeries::new(
//         sizes.iter().zip(durations.iter()).map(|(&x, &y)| (x, y)),
//         &RED,
//     ))?;    
//     chart.draw_series(LineSeries::new(
//         sizes.iter().zip(durationsna.iter()).map(|(&x, &y)| (x, y)),
//         &BLUE,
//     ))?;
//     chart.configure_series_labels().border_style(&BLACK).draw()?;
//     root_area.present()?;
//     Ok(())
// }

fn main() {
    let data = [
        0.7080797492286373,0.0482872337166802, 0.03881083077486025, 0.902054755766424


    ];

    // Create a 4x4 matrix from the provided data
    let mut a = Array2::from_shape_vec((2, 2), Vec::from(data)).expect("");
    println!("Original matrix A:\n{:?}", a);


    let eps = 0.07; // Example epsilon value for SVD computation

    // Perform SVD
    let (u, s, vt) = clear_svd(&mut a, eps);

    // Printing the original matrix A

    // Assuming s is a diagonal matrix in the output of clear_svd
    let usv = u.dot(&s).dot(&vt);

    // Print the reconstructed matrix
    println!("Reconstructed matrix U*S*Vt:\n{:?}", usv);
}