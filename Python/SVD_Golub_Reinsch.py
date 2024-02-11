# GvL    page 454 - 456  algo 8.6.2 (full svd)
import numpy as np
def house_bidiag(A):
    m,n = A.shape
    assert m >= n
    U,V = np.eye(m), np.eye(n)    
    betas_U = np.empty(n)
    betas_V = np.empty(n-1)
    for j in range(n):
        u,betas_U[j] = make_house_vec(A[j:,j])
        miniHouse = np.eye(m-j) - betas_U[j] * np.outer(u,u)
        A[j:,j:] = (miniHouse).dot(A[j:,j:])        
        A[j+1:,j] = u[1:] # [1:m-j+1]
        if j < n-1:
            v,betas_V[j] = make_house_vec(A[j,j+1:].T)
            miniHouse = np.eye(n-(j+1)) - betas_V[j] * np.outer(v,v)
            A[j:,j+1:] = A[j:, j+1:].dot(miniHouse)
            A[j ,j+2:] = v[1:] # [1:n-j]
    return betas_U, betas_V
def make_house_vec(x):
    n = x.shape[0]
    dot_1on = x[1:].dot(x[1:])
    # v is our return vector; we hack on v[0]
    v = np.copy(x)
    v[0] = 1.0
    if dot_1on < np.finfo(float).eps:
        beta = 0.0
    else:
        # apply Parlett's formula (G&vL page 210) for safe v_0 = x_0 - norm(X) 
        norm_x= np.sqrt(x[0]**2 + dot_1on)
        if x[0] <= 0:
            v[0] = x[0] - norm_x
        else:
            v[0] = -dot_1on / (x[0] + norm_x)
        beta = 2 * v[0]**2 / (dot_1on + v[0]**2)
        v = v / v[0]
    return v, beta

def extract_packed_house_bidiag(H,betas_U,betas_V):
    U  = extract_house_reflection(H, betas_U, side="lower")
    Vt = extract_house_reflection(H, betas_V, side="upper")
    B  = extract_upper_bidiag(H)
    return U, B, Vt

def extract_house_reflection(_A, betas, side="lower"):
    A, shift = (_A, 0) if side == "lower" else (_A.T,1)
    m,n = A.shape
    Q = np.eye(m)
    for j in reversed(range(min(m,n))):
        v = A[j:,j-shift].copy()
        v[0] = 1.0
        miniQ = np.eye(m-j) - betas[j-shift] * np.outer(v,v)
        Q[j:,j:] = (miniQ).dot(Q[j:,j:])
    return Q

def extract_upper_bidiag(M):
    '''from M, pull out the diagonal and superdiagonal 
       like np.triu or np.tril would  ... assume rows >= cols
       (works for non-square M)'''
    B = np.zeros_like(M)
    shape = B.shape[1]
    step = shape + 1 # # cols + 1
    end  = shape**2  # top square (shape,shape)
    B.flat[ :end:step] = np.diag(M)     # diag
    B.flat[1:end:step] = np.diag(M,+1)  # super diag
    return B


def clean_and_partition(B, eps=1e-15):
    '''return both the sub-matrix and the index it starts at'''
    remove_small(B, eps=eps)
    m,n = B.shape
    super_diag = B.flat[1::n+1]
    super_slice = find_first_run_boolean(super_diag, front=False)
    if not super_slice.stop:
        return np.empty(0), 0  
    # based on the super diagonal, we want 
    full_slice = slice(super_slice.start, super_slice.stop + 1)
    return B[full_slice, full_slice], super_slice.start


def remove_small(B, eps):
    m,n = B.shape
    diag          = B.flat[ ::n+1] # np.diag is fine, but want to explain B.flat below
    abs_diag      = np.abs(diag)
    B.flat[::n+1] = np.where(abs_diag < eps, 0.0, diag)
    super_diag     = B.flat[1::n+1]
    # superdiag -> 0 if superdiag <= eps * (this row diag + next row diag)
    # where eps is small multiple of unit roundoff
    small_val = eps * (abs_diag[:-1] + abs_diag[1:]) # or just = eps
    B.flat[1::n+1] = np.where(np.abs(super_diag) < small_val, 0.0, super_diag)
    
def find_first_run_boolean(vec, front=True):
    ''' use front=False if you will want a run from the end'''
    lbl = range(len(vec))
    if not front:
        vec = reversed(vec)
        lbl = reversed(lbl)
    
    foundStart = False
    for idx, itm in zip(lbl, vec):
        if foundStart:
            if itm:
                end = idx
            else:
                break
        else:
            if itm:
                start = end = idx
                foundStart = True
    if not foundStart:
        return slice(0,0) # equiv to slice(None, 0) [empty]
    if not front:
        start, end = end, start # if we counted from end, swap
    return slice(start, end+1)  # allow for slicing use
def decouple_or_reduce(B, zero_idxs):
    # see GvL, pg. 454 and pray for mojo
    m,n = B.shape
    lefts, rights = [], []
    for kk in zero_idxs:
        if kk < n-1:
            curr_lefts = walk_blemish_out_right(B, kk, kk+1) # decouple
            lefts.extend(curr_lefts)
        elif kk == n-1:
            rights = walk_blemish_out_up(B, kk-1, kk)    # reduce
    return lefts, rights
def walk_blemish_out_right(B, row, start_col): # usually start_col = row+1
    m,n = B.shape
    lefts = []
    for col in range(start_col, n):
        curr_left = givens_zero_wdiag_left_on_BBM(B, row, col)
        lefts.append(curr_left)
    return lefts
            
def walk_blemish_out_up(B, start_row, col): # usually start_row = col-1
    m, n = B.shape
    rights = []  # Initialize a list to collect Givens rotations
    for row in reversed(range(start_row+1)):
        # Update to collect the Givens rotation generated at each step
        givens_rotation = givens_zero_wdiag_right_on_BBM(B, row, col)
        if givens_rotation:  # Ensure the rotation is valid/non-empty
            rights.append(givens_rotation)
    return rights  # Return the collected Givens rotations

def givens_zero_wdiag_left_on_BBM(BBM, row, col):
    #assert really BBM
    m,n = BBM.shape
    c,s = zeroing_givens_coeffs(BBM[col,col], BBM[row,col])
    old_blemish = BBM[row,col]
    BBM[row,col] = 0                                  # from zeroing row
    BBM[col,col] = BBM[col,col] * c - old_blemish * s # from top row
    if col < n-1: # update these in all but last column
        BBM[row, col+1] = s * BBM[col, col+1]
        BBM[col, col+1] *= c
    return (c,s,col,row) # == G      

def givens_zero_wdiag_right_on_BBM(BBM, row, col):
    #assert really BBM
    m,n = BBM.shape
    c,s = zeroing_givens_coeffs(BBM[row,row], BBM[row,col])
    old_blemish = BBM[row,col]
    BBM[row,col] = 0                                  # from zeroing col
    BBM[row,row] = BBM[row,row] * c - old_blemish * s # from left col
    if row > 0: # update these in all but last row
        BBM[row-1, col] = s * BBM[row-1, row]
        BBM[row-1, row] *= c
    return (c,s,row,col) # == G


def step_full_bidiagonal_towards_diagonal(B):
    lastCols = B[:, -2:]
    T = lastCols.T.dot(lastCols)
    # compute wilkinson shift value ("mu")
    d  = (T[-2,-2] - T[-1,-1]) / 2.0
    shift = T[-1,-1] - T[-1,-2]**2 / (d + np.sign(d)*np.sqrt(d**2+T[-1,-2]**2))
    # special case givens_zero_right(B) on shift
    # the only explict part of T we need in this reduction step
    T_00 = np.sum(B.T[0,:]**2)
    T_01 = np.dot(B.T[0,:], B[:,1]) #B^T B --> B.T[0,:], B[:,1] 
    c,s = zeroing_givens_coeffs(T_00-shift, T_01)
    local_right = (c,s,0,1) # store for later
    right_givens(B, local_right)
    lefts, rights = bounce_blemish_down_diagonal(B)
    rights = [local_right] + rights # BONUS:  remove the copy (deque or reverse order)
    return lefts, rights

def zeroing_givens_coeffs(x,z):
    if z == 0.0: # better to check for "small": abs(z) < np.finfo(np.double).eps
        return 1.0,0.0
    r = np.hypot(x,z) # C99 hypot is safe for under/overflow
    return x/r, -z/r

    
def bounce_blemish_down_diagonal(BBM):
    m,n = BBM.shape
    lefts, rights = [], []
    for k in range(n-2):
        leftG  = givens_zero_adjacent_onband_left( BBM, k+1, k)
        rightG = givens_zero_adjacent_onband_right(BBM, k,   k+2)
        lefts.append(leftG)
        rights.append(rightG)
    leftG = givens_zero_adjacent_onband_left(BBM, n-1, n-2)
    lefts.append(leftG)
    return lefts, rights

def givens_zero_adjacent_onband_left(A, row, col):
    # affecting two rows: need the row above me
    # assert col==row-1 ... below diag ... row>col
    c,s = zeroing_givens_coeffs(A[row-1,col], A[row,col]) # same col
    G = (c,s,row-1,row)
    left_givensT(G, A) # G tells affected *rows* - one col gets 0
    return G
def givens_zero_adjacent_onband_right(A, row, col):
    # affecting two cols:  need col left of me
    # assert col-2==row ... above diag ... col>row
    c,s = zeroing_givens_coeffs(A[row,col-1], A[row,col]) # same row
    G = (c,s,col-1,col)
    right_givens(A, G) # G tells affected *cols* - one row gets 0
    return G

def apply_givens_lefts_on_right(U, lefts, shift_dex):
    for curr_left in lefts:
        c,s,c1,c2 = curr_left
        c1 += shift_dex; c2 += shift_dex
        right_givens(U, (c,s,c1,c2))
        
def right_givens(A, c_s_c1_c2):
    c,s,c1,c2=c_s_c1_c2
    givens = np.array([[ c, s],
                       [-s, c]])
    A[:,[c1, c2]] = np.dot(A[:,[c1, c2]], givens)
    
def apply_givens_rights_on_left(rights, Vt, shift_dex):
    for curr_right in rights:
        c, s, r1, r2 = curr_right  # Directly unpack the tuple without using set()
        r1 += shift_dex
        r2 += shift_dex
        left_givensT((c, s, r1, r2), Vt)  # Pass the tuple correctly to left_givensT
        
def left_givensT(c_s_r1_r2, A):
    c,s,r1,r2=c_s_r1_r2
    givensT = np.array([[ c, -s],   # manually transposed 
                        [ s,  c]])
    A[[r1,r2],:] = np.dot(givensT, A[[r1,r2],:])
    
def clear_svd(A, eps=1e-15): 
    m,n = A.shape
    betas_U, betas_V = house_bidiag(A) # betas for U and Vt
    U, B, V = extract_packed_house_bidiag(A, betas_U, betas_V)
    Vt = V.T 
    B22, k = clean_and_partition(B,eps=eps)
    while B22.size:
        zero_idxs = np.where(np.abs(np.diag(B22)) < eps)[0]
        if zero_idxs.size:
            lefts, rights = decouple_or_reduce(B22, zero_idxs)
        else:
            lefts, rights = step_full_bidiagonal_towards_diagonal(B22)
        
        apply_givens_lefts_on_right(U, lefts,   shift_dex=k)
        apply_givens_rights_on_left(rights, Vt, shift_dex=k)
        B22, k = clean_and_partition(B,eps=eps)
    return U, B, Vt
if __name__ == '__main__':
    A = np.random.random((4, 4))
    U, B, Vt = clear_svd(A.copy(), eps=1e-15)
    print("U:\n", U)
    print("S:\n", B)
    print("V^T:\n", Vt)

    
