//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap_jvc_dense.cpp
/// \brief      Решение задачи о назначениях методом JVC для плотных матриц (cтандартная линейная дискретная оптимизационная задача)
/// \date       18.05.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <lap.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{

//const double LARGE = 1.0e15;
//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_ccrrt_sparse_( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, int *free_rows, int *x, int *y, double *v, double maxcost, double resolution )
{
    int n_free_rows = 0;
    bool *unique = new bool[n];
//    memset( unique, 1, n ); // true
    for( unsigned i = 0; i < n; i++ ) {
        unique[i] = true;
    }

    for( unsigned i = 0; i < n; i++ ) {
        x[i] = -1;
        v[i] = maxcost;
        y[i] = 0;
    }
    for( unsigned i = 0; i < n; i++ ) {
        for( unsigned k = ii[i]; k < ii[i+1]; k++ ) {
            const int j = kk[k];
            const double c = cc[k];
            if( c < v[j] ) {
                v[j] = c;
                y[j] = i;
            }
//            PRINTF("i=%d, k=%d, j=%d, c[i,j]=%f, v[j]=%f y[j]=%d\n", i, k, j, c, v[j], y[j]);
        }
    }
//    PRINT_COST_ARRAY(v, n);
//    PRINT_INDEX_ARRAY(y, n);
    {
        int j = n;
        do {
            j--;
            const int i = y[j];
            if( x[i] < 0 ) {
                x[i] = j;
            } else {
                unique[i] = false;
                y[j] = -1;
            }
        } while( j > 0 );
    }
    for( unsigned i = 0; i < n; i++) {
        if( x[i] < 0 ) {
            free_rows[n_free_rows++] = i;
        } else if( unique[i] && ( ( ii[i+1] - ii[i] ) > 1 ) ) {
            const int j = x[i];
            double min = maxcost;
            for( unsigned k = ii[i]; k < ii[i+1]; k++ ) {
                const int j2 = kk[k];
                if( j2 == j ) {
                    continue;
                }
                const double c = ( cc[k] - v[j2] );
                if( c < min ) {
                    min = c;
                }
            }
//            PRINTF("v[%d] = %f - %f\n", j, v[j], min);
            v[j] -= min;
        }
    }
    delete[] unique;
    return n_free_rows;
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_carr_sparse( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, const unsigned n_free_rows, int *free_rows, int *x, int *y, double *v,
    double maxcost, double resolution )
{
    unsigned current = 0;
    int new_free_rows = 0;
    unsigned rr_cnt = 0;
//    PRINT_INDEX_ARRAY(x, n);
//    PRINT_INDEX_ARRAY(y, n);
//    PRINT_COST_ARRAY(v, n);
//    PRINT_INDEX_ARRAY(free_rows, n_free_rows);
    while( current < n_free_rows ) {
        int i0;
        int j1, j2;
        double v1, v2, v1_new;
        bool v1_lowers;

        rr_cnt++;
//        PRINTF("current = %d rr_cnt = %d\n", current, rr_cnt);
        const int free_i = free_rows[current++];
        if( ii[free_i+1] - ii[free_i] > 0 ) {
            const unsigned k = ii[free_i];
            j1 = kk[k];
            v1 = cc[k] - v[j1];
        } else {
            j1 = 0;
            v1 = maxcost;
        }
        j2 = -1;
        v2 = maxcost;
//        for( unsigned k = ii[free_i] + 1; k < ii[free_i + 1]; k++ ) {
        for( unsigned k = ii[free_i]; k < ii[free_i + 1]; k++ ) { // MY VERSION
//            PRINTF("%d = %f %d = %f\n", j1, v1, j2, v2);
            const int j = kk[k];
            const double c = cc[k] - v[j];
            if( c < v2 ) { // COMPARE
//                if( c >= v1 ) { // ORIGINAL
                if( ( c > v1 ) || ( std::abs( c - v1 ) < resolution ) ) { // MY VERSION
                    v2 = c;
                    j2 = j;
                } else {
                    v2 = v1;
                    v1 = c;
                    j2 = j1;
                    j1 = j;
                }
            }
        }
        i0 = y[j1];
        v1_new = v[j1] - ( v2 - v1 );
        v1_lowers = v1_new < v[j1];
//        PRINTF("%d %d 1=%d,%f 2=%d,%f v1'=%f(%d,%g) \n", free_i, i0, j1, v1, j2, v2, v1_new, v1_lowers, v[j1] - v1_new);
        if( rr_cnt < current * n ) {
            if( v1_lowers ) {
                v[j1] = v1_new;
            } else if( i0 >= 0 && j2 >= 0 ) {
                j1 = j2;
                i0 = y[j2];
            }
            if( i0 >= 0 ) {
                if( v1_lowers ) {
                    free_rows[--current] = i0;
                } else {
                    free_rows[new_free_rows++] = i0;
                }
            }
        } else {
//            PRINTF("rr_cnt=%d >= %d (current=%d * n=%d)\n", rr_cnt, current * n, current, n);
            if( i0 >= 0 ) {
                free_rows[new_free_rows++] = i0;
            }
        }
        x[free_i] = j1;
        y[j1] = free_i;
    }
    return new_free_rows;
}

//----------------------------------------------------------------------------------------------------------------------
unsigned jvc_sparse_find_sparse_1( const unsigned n, unsigned lo, double *d, int *cols, int *y, double resolution )
{
    unsigned hi = lo + 1;
    double mind = d[cols[lo]];
    for( unsigned k = hi; k < n; k++ ) {
        int j = cols[k];
//        if( d[j] <= mind ) { // COMPARE
        if( ( d[j] < mind ) || ( std::abs( d[j] - mind ) < resolution ) ) { // COMPARE
            if( d[j] < mind ) {
                hi = lo;
                mind = d[j];
            }
            cols[k] = cols[hi];
            cols[hi++] = j;
        }
    }
    return hi;
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_find_sparse_2( double *d, int *scan, const unsigned n_todo, int *todo, bool *done, double maxcost, double resolution )
{
    int hi = 0;
    double mind = maxcost;
    for( unsigned k = 0; k < n_todo; k++) {
        int j = todo[k];
        if( done[j] ) {
            continue;
        }
//        if( d[j] <= mind ) { // COMPARE
        if( ( d[j] < mind ) || ( std::abs( d[j] - mind ) < resolution ) ) { // COMPARE
            if( d[j] < mind ) {
                hi = 0;
                mind = d[j];
            }
            scan[hi++] = j;
        }
    }
    return hi;
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_scan_sparse_1( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, unsigned *plo, unsigned *phi, double *d, int *cols, int *pred, int *y, double *v, double resolution )
{
    unsigned lo = *plo;
    unsigned hi = *phi;
    double h, cred_ij;

    int *rev_kk = new int[n];
//    int_t *rev_kk;
//    NEW(rev_kk, int_t, n);

    while( lo != hi ) {
        int kj;
        int j = cols[lo++];
        const int i = y[j];
        const double mind = d[j];
        for( unsigned k = 0; k < n; k++ ) {
            rev_kk[k] = -1;
        }
        for( unsigned k = ii[i]; k < ii[i + 1]; k++ ) {
            const int j = kk[k];
            rev_kk[j] = k;
        }
//        PRINTF("?%d kk[%d:%d]=", j, ii[i], ii[i+1]);
//        PRINT_INDEX_ARRAY(kk + ii[i], ii[i+1] - ii[i]);
        kj = rev_kk[j];
        if( kj == -1 ) {
            continue;
        }
        assert( static_cast<int>( kk[kj] ) == j );
        h = cc[kj] - v[j] - mind;
//        PRINTF("i=%d j=%d kj=%d h=%f\n", i, j, kj, h);
        // For all columns in TODO
        for( unsigned k = hi; k < n; k++ ) {
            j = cols[k];
//            PRINTF("?%d kk[%d:%d]=", j, ii[i], ii[i+1]);
//            PRINT_INDEX_ARRAY(kk + ii[i], ii[i+1] - ii[i]);
            if( ( kj = rev_kk[j] ) == -1 ) {
                continue;
            }
            assert( static_cast<int>( kk[kj] ) == j );
            cred_ij = cc[kj] - v[j] - h;
            if( cred_ij < d[j] ) { // COMPARE
                d[j] = cred_ij;
                pred[j] = i;
//                if( cred_ij == mind ) {
                if( std::abs( cred_ij - mind ) < resolution ) {
                    if( y[j] < 0 ) {
//                        FREE(rev_kk);
                        delete[] rev_kk;
                        return j;
                    }
                    cols[k] = cols[hi];
                    cols[hi++] = j;
                }
            }
        }
    }
    *plo = lo;
    *phi = hi;
//    FREE(rev_kk);
    delete[] rev_kk;
    return -1;
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_scan_sparse_2( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, unsigned *plo, unsigned *phi, double *d, int *pred, bool *done,
    unsigned *pn_ready, int *ready, int *scan, unsigned *pn_todo, int *todo, bool *added, int *y, double *v, double resolution )
{
    unsigned lo = *plo;
    unsigned hi = *phi;
    unsigned n_todo = *pn_todo;
    unsigned n_ready = *pn_ready;
    double h, cred_ij;

    int *rev_kk = new int[n];

    for( unsigned k = 0; k < n; k++ ) {
        rev_kk[k] = -1;
    }
    while( lo != hi ) {
        int kj;
        int j = scan[lo++];
        const int i = y[j];
        ready[n_ready++] = j;
        const double mind = d[j];
        for( unsigned k = ii[i]; k < ii[i+1]; k++ ) {
            const int j = kk[k];
            rev_kk[j] = k;
        }
//        PRINTF("?%d kk[%d:%d]=", j, ii[i], ii[i+1]);
//        PRINT_INDEX_ARRAY(kk + ii[i], ii[i+1] - ii[i]);
        kj = rev_kk[j];
        assert( kj != -1 );
        assert( static_cast<int>( kk[kj] ) == j );
        h = cc[kj] - v[j] - mind;
//        PRINTF("i=%d j=%d kj=%d h=%f\n", i, j, kj, h);
        // For all columns in TODO
        for( unsigned k = 0; k < ii[i+1] - ii[i]; k++ ) {
            j = kk[ii[i] + k];
            if( done[j] ) {
                continue;
            }
//            PRINTF("?%d kk[%d:%d]=", j, ii[i], ii[i+1]);
//            PRINT_INDEX_ARRAY(kk + ii[i], ii[i+1] - ii[i]);
            cred_ij = cc[ii[i] + k] - v[j] - h;
            if( cred_ij < d[j] ) { // COMPARE
                d[j] = cred_ij;
                pred[j] = i;
//                if( cred_ij <= mind ) { // COMPARE
                if( ( cred_ij < mind ) || ( std::abs( cred_ij - mind ) < resolution ) ) { // COMPARE
                    if( y[j] < 0 ) {
//                        FREE(rev_kk);
                        delete[] rev_kk;
                        return j;
                    }
                    scan[hi++] = j;
                    done[j] = true;//TRUE;
                } else if( !added[j] ) {
                    todo[n_todo++] = j;
                    added[j] = true;//TRUE;
                }
            }
        }
        for( unsigned k = ii[i]; k < ii[i+1]; k++ ) {
            const int j = kk[k];
            rev_kk[j] = -1;
        }
    }
    *pn_todo = n_todo;
    *pn_ready = n_ready;
    *plo = lo;
    *phi = hi;
    delete[] rev_kk;
//    FREE(rev_kk);
    return -1;
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_find_path_sparse_1( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, const int start_i, int *y, double *v, int *pred, double maxcost, double resolution )
{
    unsigned lo = 0, hi = 0;
    int final_j = -1;
    unsigned n_ready = 0;
    int *cols = new int[n];
    double *d = new double[n];
//    NEW(cols, int_t, n);
//    NEW(d, cost_t, n);

    for( unsigned i = 0; i < n; i++ ) {
        cols[i] = i;
        d[i] = maxcost;
        pred[i] = start_i;
    }
    for( unsigned i = ii[start_i]; i < ii[start_i + 1]; i++ ) {
        const int j = kk[i];
        d[j] = cc[i] - v[j];
    }
//    PRINT_COST_ARRAY(d, n);
    while( final_j == -1 ) {
        // No columns left on the SCAN list.
        if( lo == hi ) {
//            PRINTF("%d..%d -> find\n", lo, hi);
            n_ready = lo;
            hi = jvc_sparse_find_sparse_1( n, lo, d, cols, y, resolution );
//            PRINTF("check %d..%d\n", lo, hi);
//            PRINT_INDEX_ARRAY(cols, n);
            for( unsigned k = lo; k < hi; k++ ) {
                const int j = cols[k];
                if( y[j] < 0 ) {
                    final_j = j;
                }
            }
        }
        if( final_j == -1 ) {
//            PRINTF("%d..%d -> scan\n", lo, hi);
            final_j = jvc_sparse_scan_sparse_1( n, cc, ii, kk, &lo, &hi, d, cols, pred, y, v, resolution );
//            PRINT_COST_ARRAY(d, n);
//            PRINT_INDEX_ARRAY(cols, n);
//            PRINT_INDEX_ARRAY(pred, n);
        }
    }

//    PRINTF("found final_j=%d\n", final_j);
//    PRINT_INDEX_ARRAY(cols, n);
    {
        const double mind = d[cols[lo]];
        for( unsigned k = 0; k < n_ready; k++ ) {
            const int j = cols[k];
            v[j] += d[j] - mind;
        }
    }

//    FREE(cols);
//    FREE(d);
    delete[] cols;
    delete[] d;

    return final_j;
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_find_path_sparse_2( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, const int start_i, int *y, double *v, int *pred, double maxcost, double resolution )
{
    unsigned lo = 0, hi = 0;
    int final_j = -1;
    unsigned n_ready = 0;
    unsigned n_todo = ( ii[start_i + 1] - ii[start_i] );

    bool *done = new bool[n];
    bool *added = new bool[n];
    int *ready = new int[n];
    int *scan = new int[n];
    int *todo = new int[n];
    double *d = new double[n];

//    memset( done, false, n );
//    memset( added, false, n );
    for( unsigned i = 0; i < n; i++ ) {
        done[i] = false;
        added[i] = false;
        // my
        ready[i] = 0;
        scan[i] = 0;
        todo[i] = 0;
    }

    for( unsigned i = 0; i < n; i++ ) {
        d[i] = maxcost;
        pred[i] = start_i;
    }
    for( unsigned i = ii[start_i]; i < ii[start_i + 1]; i++ ) {
        const int j = kk[i];
        d[j] = cc[i] - v[j];
        todo[i - ii[start_i]] = j;
        added[j] = true;//TRUE;
    }
//    PRINT_COST_ARRAY(d, n);
//    PRINT_INDEX_ARRAY(pred, n);
//    PRINT_INDEX_ARRAY(done, n);
//    PRINT_INDEX_ARRAY(ready, n_ready);
//    PRINT_INDEX_ARRAY(scan + lo, hi - lo);
//    PRINT_INDEX_ARRAY(todo, n_todo);
//    PRINT_INDEX_ARRAY(added, n);
    while( final_j == -1 ) {
        // No columns left on the SCAN list.
        if( lo == hi ) {
//            PRINTF("%d..%d -> find\n", lo, hi);
            lo = 0;
            hi = jvc_sparse_find_sparse_2( d, scan, n_todo, todo, done, maxcost, resolution );
//            PRINTF("check %d..%d\n", lo, hi);
            if( !hi ) {
                // XXX: the assignment is unsolvable, lets try to return
                // something reasonable nevertheless.
                for( unsigned j = 0; j < n; j++ ) {
                    if( !done[j] && y[j] < 0 ) {
                        final_j = j;
                    }
                }
                assert( final_j != -1 );
                break;
            }
            assert( hi > lo );
            for( unsigned k = lo; k < hi; k++ ) {
                const int j = scan[k];
                if( y[j] < 0 ) {
                    final_j = j;
                } else {
                    done[j] = true;//TRUE;
                }
            }
        }
        if( final_j == -1 ) {
//            PRINTF("%d..%d -> scan\n", lo, hi);
//            PRINT_INDEX_ARRAY(done, n);
//            PRINT_INDEX_ARRAY(ready, n_ready);
//            PRINT_INDEX_ARRAY(scan + lo, hi - lo);
//            PRINT_INDEX_ARRAY(todo, n_todo);
            final_j = jvc_sparse_scan_sparse_2( n, cc, ii, kk, &lo, &hi, d, pred, done, &n_ready, ready, scan,
                    &n_todo, todo, added, y, v, resolution );
//            PRINT_COST_ARRAY(d, n);
//            PRINT_INDEX_ARRAY(pred, n);
//            PRINT_INDEX_ARRAY(done, n);
//            PRINT_INDEX_ARRAY(ready, n_ready);
//            PRINT_INDEX_ARRAY(scan + lo, hi - lo);
//            PRINT_INDEX_ARRAY(todo, n_todo);
//            PRINT_INDEX_ARRAY(added, n);
        }
    }

//    PRINTF("found final_j=%d\n", final_j);
    {
        const double mind = d[scan[lo]];
        for( unsigned k = 0; k < n_ready; k++ ) {
            const int j = ready[k];
            v[j] += d[j] - mind;
        }
    }

//    FREE(done);
//    FREE(added);
//    FREE(ready);
//    FREE(scan);
//    FREE(todo);
//    FREE(d);
    delete[] done;
    delete[] added;
    delete[] ready;
    delete[] scan;
    delete[] todo;
    delete[] d;

    return final_j;
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_find_path_sparse_dynamic( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, const int start_i, int *y, double *v, int *pred, double maxcost, double resolution )
{
    const unsigned n_i = ii[start_i+1] - ii[start_i];
    // XXX: wouldnt it be better to decide for the whole matrix?
    if( n_i > 0.25 * n ) {
        return jvc_sparse_find_path_sparse_1( n, cc, ii, kk, start_i, y, v, pred, maxcost, resolution );
    } else {
        return jvc_sparse_find_path_sparse_2( n, cc, ii, kk, start_i, y, v, pred, maxcost, resolution );
    }
}

typedef int (*fp_function_t)( const unsigned, const std::vector<double> &, const std::vector<int> &,
    const std::vector<int> &, const int, int *, double *, int *, double, double );

fp_function_t jvc_sparse_get_better_find_path( const unsigned n, const std::vector<int> &ii )
{
    const double sparsity = ii[n] / (double)(n * n);
    if( sparsity > 0.25 ) {
//        PRINTF("Using find_path_sparse_1 for sparsity=%f\n", sparsity);
        return jvc_sparse_find_path_sparse_1;
    } else {
//        PRINTF("Using find_path_sparse_2 for sparsity=%f\n", sparsity);
        return jvc_sparse_find_path_sparse_2;
    }
}

//----------------------------------------------------------------------------------------------------------------------
int jvc_sparse_ca_sparse( const unsigned n, const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, const unsigned n_free_rows, int *free_rows, int *x, int *y,
    double *v, int fp_version, double maxcost, double resolution )
{
    int *pred = new int[n];

    fp_function_t fp;
    switch( fp_version ) {
        case FP_1: fp = jvc_sparse_find_path_sparse_1; break;
        case FP_2: fp = jvc_sparse_find_path_sparse_2; break;
        case FP_DYNAMIC: fp = jvc_sparse_get_better_find_path( n, ii ); break;
        default: return -2;
    }

    for( int *pfree_i = free_rows; pfree_i < free_rows + n_free_rows; pfree_i++ ) {
        int i = -1, j;
        unsigned k = 0;

//        PRINTF("looking at free_i=%d\n", *pfree_i);
        j = fp( n, cc, ii, kk, *pfree_i, y, v, pred, maxcost, resolution );
        assert( j >= 0 );//ASSERT(j >= 0);
        assert( j < static_cast<int>( n ) );//ASSERT(j < n);
        while( i != *pfree_i ) {
//            PRINTF("augment %d\n", j);
//            PRINT_INDEX_ARRAY(pred, n);
            i = pred[j];
//            PRINTF("y[%d]=%d -> %d\n", j, y[j], i);
            y[j] = i;
//            PRINT_INDEX_ARRAY(x, n);
//            SWAP_INDICES(j, x[i]);            
            std::swap( j, x[i] );
//            int _temp_index = j;
//            j = x[i];
//            x[i] = _temp_index;
            k++;                       

            if( k >= n ) {
                //assert( false ); // ORIGINAL
                break;
            }
        }
    }
    delete[] pred;//FREE(pred);
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------
//void JVCsparse(const std::vector<double> &cc, const std::vector<unsigned> &kk, const std::vector<unsigned> &ii,
//    unsigned dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost, TFindPath fp )
void JVCsparse(const std::vector<double> &cc, const std::vector<int> &kk, const std::vector<int> &ii,
    unsigned dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost, TFindPath fp )
{
    // Если ищем максимум - умножим матрицу на -1
    std::vector<double> cc_ = cc;
    if( sp == TSearchParam::SP_Max ) { // Поиск минимума/максимума
        std::transform( cc_.begin(), cc_.end(), cc_.begin(),
            std::bind( std::multiplies<double>(), std::placeholders::_1, -1.0 ) );
    }

    int *x = new int[dim];
    int *y = new int[dim];
    int *free_rows = new int[dim];
    double *v = new double[dim];

    int ret = jvc_sparse_ccrrt_sparse_( dim, cc_, ii, kk, free_rows, x, y, v, maxcost, resolution );
    int i = 0;
    while( ret > 0 && i < 2 ) {
        ret = jvc_sparse_carr_sparse( dim, cc_, ii, kk, ret, free_rows, x, y, v, maxcost, resolution );
        i++;
    }
    if( ret > 0 ) {
        ret = jvc_sparse_ca_sparse( dim, cc_, ii, kk, ret, free_rows, x, y, v, fp, maxcost, resolution );
    }

    // Prapare output - rowsol and lapcost.
    lapcost = 0.0;
    for( unsigned i = 0; i < dim; i++ ) { // i - row index
        rowsol[i] = x[i];
        const unsigned j_ = rowsol[i]; // j - col index
        const int start = ii[i];
        const int end = ii[i+1];
        for( int j = start; j < end; j++ ) {
            if( j_ == kk[j] ) {
                lapcost += cc[j];
                break;
            }
        }
    }
    delete[] free_rows;
    delete[] v;
    delete[] x;
    delete[] y;
//    return ret;
}

void JVCsparse( const SPML::Sparse::CMatrixCSR<double> &csr, unsigned dim, TSearchParam sp, double maxcost,
    double resolution, arma::ivec &rowsol, double &lapcost, TFindPath fp )
{
    JVCsparse( csr.csr_val, csr.csr_first, csr.csr_kk, dim, sp, maxcost, resolution, rowsol, lapcost, fp );
}

} // end namespace LAP
} // end namespace SPML
/// \}

