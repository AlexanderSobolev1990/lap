//----------------------------------------------------------------------------------------------------------------------
///
/// \file       jvcsparse.cpp
/// \brief      Решение задачи о назначениях методом JVC для разреженных матриц
/// \addtogroup spm
/// \{
///

#include "spm/lap/jvcsparse.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SPM /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{
namespace {
//----------------------------------------------------------------------------------------------------------------------
void updateDual( int nc, arma::vec &d, arma::vec &v, arma::ivec &todo, int last, double min_ )
{
    for( int k = last; k < nc; k++ ) {
        int j0 = todo(k);
        v(j0) += ( d(j0) - min_ );
    }
}

//----------------------------------------------------------------------------------------------------------------------
void updateAssignments( arma::ivec &lab, arma::ivec &y, arma::ivec &x, int j, int i0 )
{
    int tmp;
    while( true ) {
        int i = lab(j);
        y(j) = i;
        //(j, x[i]) = (x[i], j);
        tmp = j;
        j = x[i];
        x[i] = tmp;
        if( i == i0 ) {
            return;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
int solveForOneL( std::vector<double> &cc_, const std::vector<int> &kk, const std::vector<int> &first,
    int l, int nc, arma::vec &d, arma::ivec &ok, arma::ivec &free, arma::vec &v, arma::ivec &lab, arma::ivec &todo,
    arma::ivec &y, arma::ivec &x, int td1, double resolution, double infValue, bool &fail )
{
    for( int jp = 0; jp < nc; jp++ ) {
        d(jp) = infValue;
        ok(jp) = 0; // false
    }
    double min_ = infValue;
    td1 = -1;
    int i0 = free(l);
    int j;
    for( int t = first[i0]; t < first[i0 + 1]; t++ ) {
        j = kk[t];
        double dj = cc_[t] - v(j);
        d(j) = dj;
        lab(j) = i0;
//        if( dj <= min_ ) { //POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
//        if( ( dj < min_ ) || ( std::abs( dj - min_ ) < resolution ) ) { //POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        if( ( ( min_ - dj ) > resolution ) || ( std::abs( dj - min_ ) <= resolution ) ) { //POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
            //if( dj < min_ ) {
            if ( ( min_ - dj ) > resolution ) {
                td1 = -1;
                min_ = dj;
            }
            todo(++td1) = j;
        }
    }
    for( int hp = 0; hp <= td1; hp++ ) {
        j = todo(hp);
        if( y(j) == -1 ) {
            updateAssignments( lab, y, x, j, i0 );
            return td1;
        }
        ok(j) = 1;// true
    }
    int td2 = ( nc - 1 );
    int last = nc;
    while( true ) {
        if( td1 < 0 ) {
            fail = true; // FAIL!!!
            return 1;
        }
        int j0 = todo(td1--);
        int i = y(j0);
        todo(td2--) = j0;
        int tp = first[i];
        while( tp < first[i + 1] && kk[tp] != j0 ) {
            tp++;
        }
        if( tp == first[i + 1] ) {
            fail = true;
            return 1;
        }
        double h = cc_[tp] - v(j0) - min_;
        for( int t = first[i]; t < first[i + 1]; t++ ) {
            j = kk[t];
//            if( !ok(j) ) {
            if( ok(j) == 0 ) { // if( false )
                double vj = cc_[t] - v(j) - h;
//                if( vj < d(j) ) { // POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
                if( ( d(j) - vj ) > resolution ) { // POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
                    d(j) = vj;
                    lab(j) = i;
//                    if( vj == min_ ) { // POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
                    if( std::abs( vj - min_ ) <= resolution ) {
                        if( y[j] == -1 ) {
                            updateDual( nc, d, v, todo, last, min_ );
                            updateAssignments( lab, y, x, j, i0 );
                            return td1;
                        }
                        todo(++td1) = j;
                        ok(j) = 1; // true
                    }
                }
            }
        }
        if( td1 == -1 ) {
            // The original Pascal code uses finite numbers instead of double.PositiveInfinity
            // so we need to adjust slightly here.
            min_ = infValue;
            last = td2 + 1;
            for( int jp = 0; jp < nc; jp++ ) {
//                if( ( ( d[jp] < min_ ) || ( std::abs( d[jp] - min_ ) < resolution ) ) && !ok(jp) ) {
//                if( ( ( d[jp] < min_ ) || ( std::abs( d[jp] - min_ ) < resolution ) ) && ( ok(jp) == 0 ) ) {
                if( std::isfinite( d[jp] ) && ( ( ( min_ - d[jp] ) > resolution ) || ( std::abs( d[jp] - min_ ) <= resolution ) ) && ( ok(jp) == 0 ) ) {
                    //if( d[jp] < min_ ) {
                    if( ( min_ - d[jp] ) > resolution ) {
                        td1 = -1;
                        min_ = d(jp);
                    }
                    todo(++td1) = jp;
                }
            }
            for( int hp = 0; hp <= td1; hp++ ) {
                j = todo(hp);
                if( y(j) == -1 ) {
                    updateDual( nc, d, v, todo, last, min_ );
                    updateAssignments( lab, y, x, j, i0 );
                    return td1;
                }
                ok(j) = 1;//true;
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
void validateInput( const std::vector<double> &cc, const std::vector<int> &kk,
    const std::vector<int> &first, TSearchParam sp, double infValue, double resolution )
{
    if( sp != SP_Min && sp != SP_Max ) {
        throw std::invalid_argument( "LAP::JVCsparse: invalid search parameter" );
    }
    if( !std::isfinite( infValue ) || infValue <= 0.0
        || !std::isfinite( resolution ) || resolution < 0.0 ) {
        throw std::invalid_argument( "LAP::JVCsparse: invalid infinity or resolution parameter" );
    }
    if( first.empty() || cc.size() != kk.size()
        || cc.size() > static_cast<size_t>( std::numeric_limits<int>::max() )
        || first.size() > static_cast<size_t>( std::numeric_limits<int>::max() ) ) {
        throw std::length_error( "LAP::JVCsparse: invalid CSR sizes" );
    }
    if( first.front() != 0 || first.back() != static_cast<int>( cc.size() ) ) {
        throw std::invalid_argument( "LAP::JVCsparse: invalid CSR offsets" );
    }
    for( size_t i = 1; i < first.size(); ++i ) {
        if( first[i] < first[i - 1] ) {
            throw std::invalid_argument( "LAP::JVCsparse: CSR offsets must be nondecreasing" );
        }
    }
    for( size_t t = 0; t < cc.size(); ++t ) {
        if( !std::isfinite( cc[t] ) || kk[t] < 0 || kk[t] == std::numeric_limits<int>::max() ) {
            throw std::invalid_argument( "LAP::JVCsparse: invalid CSR value or column index" );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
bool updateRowPotentials( const std::vector<double> &cc, const std::vector<int> &kk,
    const std::vector<int> &first, const arma::vec &v, arma::vec &u )
{
    for( arma::uword i = 0; i < u.n_elem; ++i ) {
        double minimum = std::numeric_limits<double>::infinity();
        for( int t = first[i]; t < first[i + 1]; ++t ) {
            minimum = std::min( minimum, cc[t] - v(kk[t]) );
        }
        if( !std::isfinite( minimum ) ) {
            return false;
        }
        u(i) = minimum;
    }
    return true;
}

//----------------------------------------------------------------------------------------------------------------------
bool initializeWarmStart( const std::vector<double> &cc, const std::vector<int> &kk,
    const std::vector<int> &first, double resolution, const arma::ivec *rowsol_init,
    arma::ivec &x, arma::ivec &y, arma::vec &u, arma::vec &v )
{
    const bool rectangular = x.n_elem < y.n_elem;
    if( rectangular ) {
        for( double &value : v ) {
            value = std::min( 0.0, value );
        }
    }
    if( !updateRowPotentials( cc, kk, first, v, u ) ) {
        return false;
    }
    const bool hasAssignment = rowsol_init && rowsol_init->n_elem == x.n_elem;
    for( arma::uword i = 0; i < x.n_elem; ++i ) {
        const arma::sword candidate = hasAssignment ? (*rowsol_init)(i) : -1;
        for( int t = first[i]; t < first[i + 1]; ++t ) {
            const int j = kk[t];
            if( hasAssignment && candidate != j ) {
                continue;
            }
            if( y(j) < 0 && std::abs( cc[t] - v(j) - u(i) ) <= resolution ) {
                x(i) = j;
                y(j) = i;
                break;
            }
        }
    }
    // Свободный столбец прямоугольной задачи должен иметь нулевой потенциал.
    // Каждый повтор освобождает хотя бы одну строку: циклов вытеснения здесь нет.
    if( rectangular ) {
        bool changed;
        do {
            for( arma::uword j = 0; j < y.n_elem; ++j ) {
                if( y(j) < 0 ) {
                    v(j) = 0.0;
                }
            }
            if( !updateRowPotentials( cc, kk, first, v, u ) ) {
                return false;
            }
            changed = false;
            for( arma::uword i = 0; i < x.n_elem; ++i ) {
                if( x(i) < 0 ) {
                    continue;
                }
                for( int t = first[i]; t < first[i + 1]; ++t ) {
                    if( kk[t] == x(i) ) {
                        if( std::abs( cc[t] - v(x(i)) - u(i) ) > resolution ) {
                            y(x(i)) = -1;
                            x(i) = -1;
                            changed = true;
                        }
                        break;
                    }
                }
            }
        } while( changed );
    }
    return true;
}

} // namespace

//----------------------------------------------------------------------------------------------------------------------
int JVCsparse( const std::vector<double> &cc, const std::vector<int> &kk, const std::vector<int> &first,
    TSearchParam sp, double infValue, double resolution, arma::ivec &rowsol, double &lapcost,
    arma::vec *u_io, arma::vec *v_io, const arma::ivec *rowsol_init )
{
    validateInput( cc, kk, first, sp, infValue, resolution );
    if( u_io && u_io == v_io ) {
        throw std::invalid_argument( "LAP::JVCsparse: u and v must be distinct vectors" );
    }
    // Конечный infValue сохранён в API, но не ограничивает цены и расстояния.
    const double infinity = std::numeric_limits<double>::infinity();
    // Объявления
    int nr = first.size() - 1; // Кол-во строк
    int max_kk = -1;
    for( unsigned i = 0; i < kk.size(); i++ ) { // Поиск максимального элемента в массиве kk - это и будет кол-во столбцов
        if( kk[i] > max_kk ) {
            max_kk = kk[i];
        }
    }
    int nc = max_kk + 1;// Кол-во столбцов
    // Повторяющиеся столбцы строки неоднозначны для задачи о назначениях.
    std::vector<int> seenColumn( nc, -1 );
    for( int i = 0; i < nr; ++i ) {
        for( int t = first[i]; t < first[i + 1]; ++t ) {
            if( seenColumn[kk[t]] == i ) {
                throw std::invalid_argument( "LAP::JVCsparse: duplicate edge in a CSR row" );
            }
            seenColumn[kk[t]] = i;
        }
    }
    if( nr > nc ) {
        return 1;
    }
    for( int i = 0; i < nr; ++i ) {
        if( first[i] == first[i + 1] ) {
            return 1;
        }
    }
    arma::ivec x = arma::ivec( nr, arma::fill::zeros );
    arma::ivec y = arma::ivec( nc, arma::fill::zeros );
    arma::vec u = arma::vec( nr, arma::fill::zeros );
    arma::vec v = arma::vec( nc, arma::fill::zeros );
    arma::vec d = arma::vec( nc, arma::fill::zeros );
    arma::ivec ok = arma::ivec( nc, arma::fill::zeros ); // bool: 0 - false, 1 - true (classical c-style)
    arma::ivec xinv = arma::ivec( nr, arma::fill::zeros ); // bool: 0 - false, 1 - true (classical c-style)
    arma::ivec free = arma::ivec( nr, arma::fill::zeros );
    arma::ivec todo = arma::ivec( nc, arma::fill::zeros );
    arma::ivec lab = arma::ivec( nc, arma::fill::zeros );
    int l0 = 0;

    x -= 1;
    y -= 1;
    free -= 1;
    todo -= 1;

    // Поиск минимума/максимума
    std::vector<double> cc_ = cc;
    if( sp == TSearchParam::SP_Max ) {
        std::transform( cc_.begin(), cc_.end(), cc_.begin(),
            std::bind( std::multiplies<double>(), std::placeholders::_1, -1.0 ) ); // Умножим на -1 для поиска максимума
    }

    // The initialization steps of LAPJVsp only make sense for square matrices
    const bool warm = u_io && v_io
        && u_io->n_elem == static_cast<arma::uword>( nr )
        && v_io->n_elem >= static_cast<arma::uword>( nc )
        && u_io->is_finite() && v_io->is_finite();
    if( warm ) {
        for( int j = 0; j < nc; ++j ) {
            v(j) = (*v_io)(j);
        }
        if( !initializeWarmStart( cc_, kk, first, resolution, rowsol_init, x, y, u, v ) ) {
            return 1;
        }
        for( int i = 0; i < nr; ++i ) {
            if( x(i) < 0 ) {
                free(l0++) = i;
            }
        }
    } else if( nr == nc ) {
        for( int jp = 0; jp < nc; jp++ ) {
            v(jp) = infinity;
        }
        for( int i = 0; i < nr; i++ ) {
            for( int t = first[i]; t < first[i + 1]; t++ ) {
                int jp = kk[t];
//                if( cc_[t] < v(jp) ) {
                if( ( v(jp) - cc_[t] ) > resolution ) {
                    v(jp) = cc_[t];
                    y(jp) = i;
                }
            }
        }
        for( int jp = ( nc - 1 ); jp >= 0; jp-- ) {
            int i = y(jp);
            if( i < 0 ) {
                return 1; // В квадратной задаче пустой столбец исключает полное назначение.
            }
            if( x(i) == -1 ) {
                x(i) = jp;
            } else {
                y(jp) = -1;
                // Here, the original Pascal code simply inverts the sign of x; as that
                // doesn't play too well with zero-indexing, we explicitly keep track of
                // uniqueness instead.
                xinv(i) = 1;
            }
        }
        int lp = 0;
        for( int i = 0; i < nr; i++ ) {
            if( xinv(i) ) {
                continue;
            }
            if( x(i) != -1 ) {
                double min_ = infinity;
                int j1 = x(i);
                for( int t = first[i]; t < first[i + 1]; t++ ) {
                    int jp = kk[t];
                    if( jp != j1 ) {
//                        if( ( cc_[t] - v(jp) ) < min_ ) {
                        if( ( min_ - ( cc_[t] - v(jp) ) ) > resolution ) {
                            min_ = ( cc_[t] - v(jp) );
                        }
                    }
                }
                u(i) = min_;
                int tp = first[i];
                while( kk[tp] != j1 ) {
                    tp++;
                }
                if( std::isfinite( min_ ) ) {
                    v(j1) = cc_[tp] - min_;
                } else {
                    u(i) = cc_[tp] - v(j1); // Единственное ребро: бесконечная редукция не нужна.
                }
            } else {
                free(lp++) = i;
            }
        }
        for( int tel = 0; tel < 2; tel++ ) {
            int h = 0;
            int l0p = lp;
            lp = 0;
            int reductions = 0;
            const int reductionLimit = std::max( 1, nr );
            while( h < l0p && reductions++ < reductionLimit ) {
                // Note: In the original Pascal code, the indices of the lowest
                // and second-lowest reduced costs are never reset. This can
                // cause issues for infeasible problems; see https://stackoverflow.com/q/62875232/5085211
                int i = free(h++);

                //------------------------------------------------------------------------------------------------------
                // ORIGINAL SEARCH OF MIN AND SUBMIN
                //------------------------------------------------------------------------------------------------------

                int j0p = -1; // Index of minimum
                int j1p = -1; // Index of subminimum
                double v0 = infinity; // Value of minimum
                double vj = infinity; // Value of subminimum
                for( int t = first[i]; t < first[i + 1]; t++ ) {
                    int jp = kk[t];
                    double dj = cc_[t] - v(jp);
//                    if( dj < vj ) {
                    if( ( vj - dj ) > resolution ) {
//                        if( dj >= v0 ) { // POSSIBLE FLOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
//                        if( ( dj > v0 ) || ( std::abs( dj - v0 ) < resolution ) ) {
                        if( ( ( dj - v0 ) > resolution ) || ( std::abs( dj - v0 ) <= resolution ) ) {
                            vj = dj;
                            j1p = jp;
                        } else {
                            vj = v0;
                            v0 = dj;
                            j1p = j0p;
                            j0p = jp;
                        }
                    }
                }
                // If the index of the column with the largest reduced cost has not been
                // set, no assignment is possible for this row.
                if( j0p < 0 ) {
                    return 1; // No feasible solution!!!
                }
                int i0 = y(j0p);
                if( j1p < 0 ) {
                    // Только одно ребро: назначаем без бесконечного изменения потенциала.
                    u(i) = v0;
                    x(i) = j0p;
                    y(j0p) = i;
                    if( i0 >= 0 ) {
                        x(i0) = -1;
                        free(lp++) = i0;
                    }
                    continue;
                }

                //------------------------------------------------------------------------------------------------------
                // MY SEARCH OF MIN AND SUBMIN
                //------------------------------------------------------------------------------------------------------
//                // find minimum and second minimum reduced cost over columns.
//                int j0p = -1; // Index of minimum
//                int j1p = -1; // Index of subminimum
//                double v0 = infinity; // Value of minimum
//                double vj = infinity; // Value of subminimum
//                arma::vec i_th_row( nc, arma::fill::zeros );
//                i_th_row.fill( infinity );
//                for( int t = first[i]; t < first[i + 1]; t++ ) {
//                    int jp = kk[t];
//                    double dj = cc_[t] - v(jp);
//                    i_th_row(jp) = dj;
//                }
//                j0p = arma::index_min( i_th_row );  // Index of minimum
//                v0 = i_th_row(j0p);                 // Value of minimum
//                i_th_row(j0p) = infinity;

//                j1p = arma::index_min( i_th_row );  // Index of subminimum
//                vj = i_th_row(j1p);                 // Value of subminimum

//                int i0 = y(j0p);
                //------------------------------------------------------------------------------------------------------
                // ORIGINAL
                //------------------------------------------------------------------------------------------------------

                u(i) = vj;
//                if( v0 < vj ) { // FIX
                if( ( vj - v0 ) > resolution ) { // MY
                    v(j0p) += ( v0 - vj );
                } else if( i0 != -1 ) {
                    j0p = j1p;
                    i0 = y(j0p);
                }
                x(i) = j0p;
                y(j0p) = i;
                if( i0 != -1 ) {
                    x(i0) = -1;
//                    if( v0 < vj ) { // FIX
                    if( ( vj - v0 ) > resolution ) { // MY
                        free(--h) = i0;
                    } else {
                        free(lp++) = i0;
                    }
                }

                //------------------------------------------------------------------------------------------------------
                // MY SEARCH OF MIN AND SUBMIN
                //------------------------------------------------------------------------------------------------------
//                u(i) = vj;
//                if( ( vj - v0 ) > resolution ) { // if( v0 < vj )
//                    // change the reduction of the minimum column to increase the minimum
//                    // reduced cost in the row to the subminimum.
//                    v(j0p) += ( v0 - vj );
//                } else {
//                    if( i0 > -1 ) { // minimum and subminimum equal.
//                        // minimum column j1 is assigned.
//                        // swap columns j1 and j2, as j2 may be unassigned.
//                        j0p = j1p;
//                        i0 = y(j0p);
//                    }
//                }
//                // (re-)assign i to j1, possibly de-assigning an i0.
//                x(i) = j0p;
//                y(j0p) = i;
//                if( i0 > -1 ) {
//                    if( ( vj - v0 ) > resolution ) { // FIX
//                        free(--h) = i0;
//                    } else {
//                        free(lp++) = i0;
//                    }
//                }
                //------------------------------------------------------------------------------------------------------
            }
            // После ограничения аукциона собираем все свободные строки, включая ещё не обработанные.
            lp = 0;
            for( int i = 0; i < nr; ++i ) {
                if( x(i) < 0 ) {
                    free(lp++) = i;
                }
            }
        } // end for( int tel = 0; tel < 2; tel++ )
        l0 = lp;
    } else { // end if( nr == nc )
        l0 = nr;
        for( int i = 0; i < nr; i++ ) {
            free(i) = i;
        }
    }
    int td1 = -1;
    for( int l = 0; l < l0; l++ ) {
        bool fail = false;
        td1 = solveForOneL( cc_, kk, first, l, nc, d, ok, free, v, lab, todo, y, x, td1, resolution, infinity, fail );
        if( fail ) {
            return 1;
        }
    }
    // Prapare output - rowsol and lapcost.
    double resultCost = 0.0;
    for( int i = 0; i < nr; i++ ) { // i - row index
        const int j_ = x[i]; // j - col index
        const int start = first[i];
        const int end = first[i+1];
        for( int j = start; j < end; j++ ) {
            if( j_ == kk[j] ) {
                resultCost += cc[j];
                u(i) = cc_[j] - v(j_);
                break;
            }
        }
    }
    rowsol = std::move( x );
    lapcost = resultCost;
    if( u_io ) {
        *u_io = std::move( u );
    }
    if( v_io ) {
        *v_io = std::move( v );
    }
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------
int JVCsparse( const Sparse::CMatrixCSR &csr, TSearchParam sp, double infValue, double resolution, arma::ivec &rowsol,
    double &lapcost, arma::vec *u, arma::vec *v, const arma::ivec *rowsol_init )
{
    int result = JVCsparse( csr.csr_val, csr.csr_kk, csr.csr_first, sp, infValue, resolution, rowsol, lapcost, u, v, rowsol_init );
    return result;
}

} // namespace LAP
} // namespace SPM
/// \}
