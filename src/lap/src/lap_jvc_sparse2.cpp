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
//int solveForOneL( std::vector<double> &cc_, const std::vector<unsigned> &kk, const std::vector<unsigned> &first,
//    int l, int nc, arma::vec &d, arma::ivec &ok, arma::ivec &free, arma::vec &v, arma::ivec &lab, arma::ivec &todo,
//    arma::ivec &y, arma::ivec &x, int td1, double resolution, double maxcost, bool &fail )
int solveForOneL( std::vector<double> &cc_, const std::vector<int> &kk, const std::vector<int> &first,
    int l, int nc, arma::vec &d, arma::ivec &ok, arma::ivec &free, arma::vec &v, arma::ivec &lab, arma::ivec &todo,
    arma::ivec &y, arma::ivec &x, int td1, double resolution, double maxcost, bool &fail )
{
    for( int jp = 0; jp < nc; jp++ ) {
        d(jp) = maxcost;
        ok(jp) = 0; // false
    }
    double min_ = maxcost;
    int i0 = free(l);
    int j;
    for( int t = first[i0]; t < first[i0 + 1]; t++ ) {
        j = kk[t];
        double dj = cc_[t] - v(j);
        d(j) = dj;
        lab(j) = i0;
//        if( dj <= min_ ) { //POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        if( ( dj < min_ ) || ( std::abs( dj - min_ ) < resolution ) ) { //POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
            if( dj < min_ ) {
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
        while( kk[tp] != j0 ) {
            tp++;
        }
        double h = cc_[tp] - v(j0) - min_;
        for( unsigned t = first[i]; t < first[i + 1]; t++ ) {
            j = kk[t];
            if( !ok(j) ) {
                double vj = cc_[t] - v(j) - h;
                if( vj < d(j) ) {
                    d(j) = vj;
                    lab(j) = i;
//                    if( vj == min_ ) { // POSSIBLE FLOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
                    if( std::abs( vj - min_ ) < resolution ) {
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
            min_ = maxcost;
            last = td2 + 1;
            for( int jp = 0; jp < nc; jp++ ) {
                if( ( ( d[jp] < min_ ) || ( std::abs( d[jp] - min_ ) < resolution ) ) && !ok(jp) ) {
                    if( d[jp] < min_ ) {
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
int JVCsparseNEW( const std::vector<double> &cc, const std::vector<int> &kk, const std::vector<int> &first,
    TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost )
{
    // Объявления
    int nr = first.size() - 1; // Кол-во строк
    int max_kk = -1;
    for( int i = 0; i < kk.size(); i++ ) { // Поиск максимального элемента в массиве kk - это и будет кол-во столбцов
        if( kk[i] > max_kk ) {
            max_kk = kk[i];
        }
    }
    int nc = max_kk + 1;// Кол-во столбцов
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
//    x.fill( -1 );
//    y.fill( -1 );
//    free.fill( -1 );
//    todo.fill( -1 );

    // Поиск минимума/максимума
    std::vector<double> cc_ = cc;
    if( sp == TSearchParam::SP_Max ) {
        std::transform( cc_.begin(), cc_.end(), cc_.begin(),
            std::bind( std::multiplies<double>(), std::placeholders::_1, -1.0 ) ); // Умножим на -1 для поиска максимума
    }

    // The initialization steps of LAPJVsp only make sense for square matrices
    if( nr == nc ) {
        for( int jp = 0; jp < nc; jp++ ) {
            v(jp) = maxcost;
        }
        for( int i = 0; i < nr; i++ ) {
            for( int t = first[i]; t < first[i + 1]; t++ ) {
                int jp = kk[t];
                if( cc_[t] < v(jp) ) {
                    v(jp) = cc_[t];
                    y(jp) = i;
                }
            }
        }
        for( int jp = ( nc - 1 ); jp >= 0; jp-- ) {
            int i = y(jp);
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
                double min_ = maxcost;
                int j1 = x(i);
                for( int t = first[i]; t < first[i + 1]; t++ ) {
                    int jp = kk[t];
                    if( jp != j1 ) {
                        if( ( cc_[t] - v(jp) ) < min_ ) {
                            min_ = ( cc_[t] - v(jp) );
                        }
                    }
                }
                u(i) = min_;
                int tp = first[i];
                while( kk[tp] != j1 ) {
                    tp++;
                }
                v(j1) = cc_[tp] - min_;
            } else {
                free(lp++) = i;
            }
        }
        for( int tel = 0; tel < 2; tel++ ) {
            int h = 0;
            int l0p = lp;
            lp = 0;
            while( h < l0p ) {
                // Note: In the original Pascal code, the indices of the lowest
                // and second-lowest reduced costs are never reset. This can
                // cause issues for infeasible problems; see https://stackoverflow.com/q/62875232/5085211
                int i = free(h++);

                //------------------------------------------------------------------------------------------------------
                // ORIGINAL SEARCH OF MIN AND SUBMIN
                //------------------------------------------------------------------------------------------------------

                int j0p = -1; // Index of minimum
                int j1p = -1; // Index of subminimum
                double v0 = maxcost; // Value of minimum
                double vj = maxcost; // Value of subminimum
                for( int t = first[i]; t < first[i + 1]; t++ ) {
                    int jp = kk[t];
                    double dj = cc_[t] - v(jp);
                    if( dj < vj ) {
//                        if( dj >= v0 ) { // POSSIBLE FLOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                        if( ( dj > v0 ) || ( std::abs( dj - v0 ) < resolution ) ) {
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

                //------------------------------------------------------------------------------------------------------
                // MY SEARCH OF MIN AND SUBMIN
                //------------------------------------------------------------------------------------------------------
//                // find minimum and second minimum reduced cost over columns.
//                int j0p = -1; // Index of minimum
//                int j1p = -1; // Index of subminimum
//                double v0 = maxcost; // Value of minimum
//                double vj = maxcost; // Value of subminimum
//                arma::vec i_th_row( nc, arma::fill::zeros );
//                i_th_row.fill( maxcost );
//                for( int t = first[i]; t < first[i + 1]; t++ ) {
//                    int jp = kk[t];
//                    double dj = cc_[t] - v(jp);
//                    i_th_row(jp) = dj;
//                }
//                j0p = arma::index_min( i_th_row );  // Index of minimum
//                v0 = i_th_row(j0p);                 // Value of minimum
//                i_th_row(j0p) = maxcost;

//                j1p = arma::index_min( i_th_row );  // Index of subminimum
//                vj = i_th_row(j1p);                 // Value of subminimum

//                int i0 = y(j0p);
                //------------------------------------------------------------------------------------------------------
                // ORIGINAL
                //------------------------------------------------------------------------------------------------------

                u(i) = vj;
                //if( v0 < vj ) { // FIX
                if( ( vj - v0 ) > resolution ) { // FIX
                    v(j0p) += ( v0 - vj );
                } else if( i0 != -1 ) {
                    j0p = j1p;
                    i0 = y(j0p);
                }
                x(i) = j0p;
                y(j0p) = i;
                if( i0 != -1 ) {
//                    if( v0 < vj ) { // FIX
                    if( ( vj - v0 ) > resolution ) { // FIX
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
        td1 = solveForOneL( cc_, kk, first, l, nc, d, ok, free, v, lab, todo, y, x, td1, resolution, maxcost, fail );
        if( fail ) {
            return 1;
        }
    }
    // Prapare output - rowsol and lapcost.
    lapcost = 0.0;
    for( int i = 0; i < nr; i++ ) { // i - row index
        rowsol[i] = x[i];
        const int j_ = rowsol[i]; // j - col index
        const int start = first[i];
        const int end = first[i+1];
        for( int j = start; j < end; j++ ) {
            if( j_ == kk[j] ) {
                lapcost += cc[j];
                break;
            }
        }
    }
    return 0;
}

/*
void JVCsparse2( const std::vector<double> &cc, const std::vector<unsigned> &first, const std::vector<unsigned> &kk,
    int dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost )
{
    std::vector<double> cc_ = cc;
    if( sp == TSearchParam::SP_Max ) { // Поиск минимума/максимума
        std::transform( cc_.begin(), cc_.end(), cc_.begin(),
            std::bind( std::multiplies<double>(), std::placeholders::_1, -1.0 ) );
    }

    bool unassignedfound;
    int i = 0, imin = 0, numfree = 0, prvnumfree = 0, f = 0, i0 = 0, k = 0, freerow = 0; // row
    int j = 0, j1 = 0, j2 = 0, endofpath = 0, last = 0, low = 0, up = 0;
    double dmin = 0.0, h = 0.0, umin = 0.0, usubmin = 0.0, v2 = 0.0, cost_i_j; // cost

    arma::vec u = arma::vec( dim, arma::fill::zeros );//new double[dim];
    arma::vec v = arma::vec( dim, arma::fill::zeros );//new double[dim];
    arma::ivec free = arma::ivec( dim, arma::fill::zeros );//new int[dim];       // list of unassigned rows.
    arma::ivec collist = arma::ivec( dim, arma::fill::zeros );//new int[dim];    // list of columns to be scanned in various ways.
    arma::ivec matches = arma::ivec( dim, arma::fill::zeros );//new int[dim];    // counts how many times a row could be assigned.
    arma::vec d = arma::vec( dim, arma::fill::zeros );//new double[dim];       // 'cost-distance' in augmenting path calculation.
    arma::ivec pred = arma::ivec( dim, arma::fill::zeros );//new int[dim];       // row-predecessor of column in augmenting/alternating path.
    arma::ivec colsol = arma::ivec( dim, arma::fill::zeros );//int *colsol = new int[dim];
    arma::vec x = arma::vec( dim, arma::fill::zeros );
    arma::vec xh = arma::vec( dim, arma::fill::zeros );
    arma::ivec vf0 = arma::ivec( dim, arma::fill::zeros );

    // init
    rowsol.zeros();
    rowsol -= 1;
    colsol -= 1;

    // COLUMN REDUCTION
    for( j = ( dim - 1 ); j >= 0; j-- ) { // reverse order gives better results.
        // find minimum cost over rows.
        dmin = cc_[kk[j]]; // dmin = cost(0,j); // нужен элемент с индексом 0 в каждой строке
        imin = 0;
        for( i = 1; i < dim; i++ ) {
            for( unsigned z = 0; z < first.size(); z++ ) {
                if( first(z) == i ) {
                    cost_i_j = cc_[z];
                    if( cost_i_j < dmin ) { // if( cost(i,j) < dmin ) {
                        dmin = cost_i_j;
                        imin = i;
                    }
                }
            }
        }
        v(j) = dmin;
        matches(imin)++;
        if( matches(imin) == 1 ) {
            // init assignment if minimum row assigned for first time.
            rowsol(imin) = j;
            colsol(j) = imin;
        } else if( v(j) < v(rowsol(imin)) ) {
            int j1 = rowsol(imin);
            rowsol(imin) = j;
            colsol(j) = imin;
            colsol(j1) = -1;
        } else {
            colsol(j) = -1; // row already assigned, column not assigned.
        }
    }

    // REDUCTION TRANSFER
    for( i = 0; i < dim; i++ ) {
        if( matches(i) == 0 ) { // fill list of unassigned 'free' rows.
            free(numfree) = i;
            numfree++;
        } else {
            if( matches(i) == 1 ) { // transfer reduction from rows that are assigned once.
                j1 = rowsol(i);
                for( j = 0; j < dim; j++ ) {
                    x(j) = cost(i,j) - v(j);
                }
                x(j1) = maxcost;
                v(j1) = v(j1) - x.min();
            }
        }
    }

    // AUGMENTING ROW REDUCTION
    int loopcnt = 0; // do-loop to be done twice.
    while( loopcnt < 2 ) {
        loopcnt++;
        //     scan all free rows.
        //     in some cases, a free row may be replaced with another one to be scanned next.
        k = 0;
        prvnumfree = numfree;
        numfree = 0;             // start list of rows still free after augmenting row reduction.
        while( k < prvnumfree ) {
            i = free(k);
            k++;
            // find minimum and second minimum reduced cost over columns.
            for( j = 0; j < dim; j++ ) {
                x(j) = cost(i,j) - v(j);
            }

            j1 = arma::index_min( x );
            umin = x(j1);
            x(j1) = maxcost;

            j2 = arma::index_min( x );
            usubmin = x(j2);

            i0 = colsol(j1);
            if( ( usubmin - umin ) > resolution ) {
                // change the reduction of the minimum column to increase the minimum
                // reduced cost in the row to the subminimum.
                v(j1) = v(j1) - ( usubmin - umin );
            } else {
                if( i0 > -1 ) { // minimum and subminimum equal.
                    // minimum column j1 is assigned.
                    // swap columns j1 and j2, as j2 may be unassigned.
                    j1 = j2;
                    i0 = colsol(j2);
                }
            }
            // (re-)assign i to j1, possibly de-assigning an i0.
            rowsol(i) = j1;
            colsol(j1) = i;
            if( i0 > -1 ) {  // minimum column j1 assigned earlier. // ORIGINAL
                //if( umin < ( usubmin + EPS ) ) { // fixed EPS
                if( ( usubmin - umin ) > resolution ) {
                    // put in current k, and go back to that k.
                    // continue augmenting path i - j1 with i0.
                    k--;
                    free(k) = i0;
                } else {
                    // no further augmenting reduction possible.
                    // store i0 in list of free rows for next phase.
                    free(numfree) = i0;
                    numfree++;
                }
            }
        }
    }

    // AUGMENT SOLUTION FOR EACH FREE ROW
    for( f = 0; f < numfree; f++ ) {
        freerow = free(f); // start row of augmenting path.

        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for( j = 0; j < dim; j++ ) {
            d(j) = cost(freerow,j) - v(j);
            pred(j) = freerow;
            collist(j) = j; // init column list.
        }
        low = 0;    // columns in 0..low-1 are ready, now none.
        up = 0;     // columns in low..up-1 are to be scanned for current minimum, now none.
                    // columns in up..dim-1 are to be considered later to find new minimum,
                    // at this stage the list simply contains all columns
        unassignedfound = false;
        while( !unassignedfound ) {
            if( up == low ) { // no more columns to be scanned for current minimum.
                last = low - 1;
                // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                dmin = d(collist(up));
                up++;
                for( k = up; k < dim; k++ ) {
                    j = collist(k);
                    h = d(j);
                    if( ( h < dmin ) || ( std::abs( h - dmin ) < resolution ) ) { //
//                    if( h <= dmin ) { // ORIGINAL
                        if( h < dmin ) {  // new minimum.
                            up = low;   // restart list at index low.
                            dmin = h;
                        }
                        // new index with same minimum, put on undex up, and extend list.
                        collist(k) = collist(up);
                        collist(up) = j;
                        up++;
                    }
                }
                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for( k = low; k < up; k++ ) {
                    if( colsol(collist(k)) < 0 ) {
                        endofpath = collist(k);
                        unassignedfound = true;
                        break;
                    }
                }
            }
            if( !unassignedfound ) {
                // update 'distances' between freerow and all unscanned columns, via next scanned column.
                j1 = collist(low);
                low++;
                i = colsol(j1);

                h = cost(i,j1) - v(j1) - dmin;
                for( k = up; k < dim; k++ ) {
                    j = collist(k);
                    v2 = cost(i,j) - v(j) - h;
                    if( v2 < d(j) ) {
                        pred(j) = i;
                        // if( v2 == min ) {  // new column found at same minimum value - ORIGINAL
                        if( std::abs( v2 - dmin ) < resolution ) {  // new column found at same minimum value - MY VERSION
                            if( colsol(j) < 0 ) {
                                // if unassigned, shortest augmenting path is complete.
                                endofpath = j;
                                unassignedfound = true;
                                break;
                            } else {
                                // else add to list to be scanned right away.
                                collist(k) = collist(up);
                                collist(up) = j;
                                up++;
                            }
                        }
                        d(j) = v2;
                    }
                }
            }
        }

        // update column prices.
        for( k = 0; k <= last; k++ ) { //for( k = last + 1; k--; ) {
            j1 = collist(k);
            v(j1) = v(j1) + d(j1) - dmin;
        }

        // reset row and column assignments along the alternating path.
        while( true ) {
            i = pred(endofpath);
            colsol(endofpath) = i;
            j1 = endofpath;
            endofpath = rowsol(i);
            rowsol(i) = j1; // Вывод результата
            if( i == freerow ) {
                break;
            }
        }
    }

    // calculate lapcost.
    lapcost = 0.0;
    for( i = 0; i < dim; i++ ) {
        j = rowsol(i);
        u[i] = cost(i,j) - v(j);
        double element_i_j = assigncost(i,j);
        if( !SPML::Compare::AreEqualAbs( element_i_j, maxcost, resolution ) ) {
            lapcost += element_i_j;
        }
    }
    return;
}
*/

/*
void JVCsparse2( const std::vector<double> &cc, const std::vector<unsigned> &kk, const std::vector<unsigned> &first,
    unsigned dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost )
{
    int n = dim;

//    INTEGER CC(2500),KK(2500),FIRST(101),X(100),Y(100),U(100),V(100)
//    INTEGER H,CNT,L0,T,T0,TD,V0,VJ,DJ
//    INTEGER LAB(100),D(100),FREE(100),TODO(100)
//    LOGICAL OK(100)

    arma::vec u = arma::vec( dim, arma::fill::zeros );//new double[dim];
    arma::vec v = arma::vec( dim, arma::fill::zeros );//new double[dim];
    arma::vec x = arma::vec( dim, arma::fill::zeros );
    arma::vec y = arma::vec( dim, arma::fill::zeros );

    arma::vec lab = arma::vec( dim, arma::fill::zeros );
    arma::vec d = arma::vec( dim, arma::fill::zeros );
    arma::ivec free = arma::ivec( dim, arma::fill::zeros );
    arma::ivec todo = arma::ivec( dim, arma::fill::zeros );
    arma::ivec ok = arma::ivec( dim, arma::fill::zeros );

    unsigned i, i0, j, t, j0, j1, l, l0, k, td;
    int td1, td2, last, t0;
    double v0, vj, h, dj;
    double min_;

    // Start
    for( j = 0; j < n; j++ ) {
        v(j) = maxcost;
    }
    for( i = 0; i < n; i++ ) {
        x(i) = -1;//0.0;
        y(i) = -1;// added!
        u(i) = 0.0;
        for( t = first[i]; t < first[i+1]; t++ ) {
            j = kk[t];
            if( cc_[t] < v(j) ) {
                v(j) = cc_[t];
                y(j) = i;
            }
        }
    } // 20 CONTINUE
    // COLUMN REDUCTION
    for( j = ( n - 1 ); j >= 0; j-- ) {
        i = y(j0);
        if( !SPML::Compare::IsZeroAbs( x(i), resolution ) ) { // x(i) != 0
            x(i) = -std::abs( x(i) );
            y(j0) = -1.0;//0.0; // fixed
        } else {
            x(i) = j0;
        }
    } // 30 CONTINUE

    l = 0;
    for( i = 0; i < n; i++ ) {
        if( SPML::Compare::IsZeroAbs( x(i), resolution ) ) { // x(i) == 0
            free(l) = i;
            l++;
            continue;
        }
        if( x(i) < 0.0 ) {
            x(i) = -x(i);
        } else {
            j1 = x(i);
            min_ = maxcost;
            for( t = first[i]; t < first[i+1]; t++ ) { // DO 31
                j = kk[t];
                if( j == j1 ) {
                    continue;
                }
                if( ( cc_[t] - v(j) ) < min_ ) {
                    min_ = ( cc_[t] - v(j) );
                }
            } // 31 CONTINUE
            v(j1) = v(j1) - min_;
        }
    } // 40 CONTINUE

    // IMPROVE THE INITIAL SOLUTION

    if( l != 0 ) {

    int cnt = 0; // do-loop to be done twice.
    while( ( cnt < 2 ) && ( l > 0 ) ) {
        cnt++;

        l0 = l;
        k = 0; // !
        l = 0;
        while( k < l0 ) {
            j0 = 0; j1 = 0; // this is new!
            i = free(k);
            k++;
            v0 = maxcost;
            vj = maxcost;
            for( t = first[i]; t < first[i+1]; t++ ) {
                j = kk[t];
                h = cc_[t] - v(j);
                if( h < vj ) {
                    if( ( h > v0 ) || ( SPML::Compare::AreEqualAbs( h, v0, resolution ) ) ) { // if( h >= v0 )
                        vj = h;
                        j1 = j;
                    } else {
                        vj = v0;
                        v0 = h;
                        j1 = j0;
                        j0 = j;
                    }
                }
            } // 42 CONTINUE
            i0 = y(j0);
            u(i) = vj;

//            if( v0 < vj ) {
//                v(j0) = v(j0) - vj + v0;
//            } else {
//                if( i0 == 0 ) goto M43;
//                j0 = j1;
//                i0 = y(j1);
//            }
//            if( i0 == 0 ) goto M43;
//            if( v0 < vj ) {
//                k--;
//                free(k) = i0;
//            } else {
//                free(l) = i0;
//                l++;
//            }
//M43:        x(i) = j0;
//            y(j0) = i;
            if( ( vj - v0 ) > resolution ) { //if( v0 < vj ) {
                v(j0) = v(j0) - ( vj - v0 );
            } else {
                if( i0 > -1 ) {
                    j0 = j1;
                    i0 = y(j1);
                }
            }
            x(i) = j0;
            y(j0) = i;
            if( i0 > -1 ) {
                if( ( vj - v0 ) > resolution ) {
                    k--;
                    free(k) = i0;
                } else {
                    free(l) = i0;
                    l++;
                }
            }
        }
    }

    // AUGMENTATION PART
    h = 0;
    for( i = 0; i < n; i++ ) {
        h += ( u(i) + v(i) );
    }
    l0 = l;
    for( l = 0; l < l0; l++ ) {
        for( j = 0; j < n; j++ ) {
            ok(j) = false;
            d(j) = maxcost;
        }
        min_ = maxcost;
        i0 = free(l);
//        td = n;
        for( t = first[i0]; t < first[i0+1]; t++ ) { // DO 52
            j = kk[t];
            dj = cc_[t] - v(j);
            d(j) = dj;
            lab(j) = i0;
            if( ( dj < min_ ) || ( SPML::Compare::AreEqualAbs( dj, min_, resolution ) ) ) { // if( dj <= min )
                if( dj < min_ ) {
                    min_ = dj;
                    td1 = 0;
//                    k = 0; // 1 ?????????
//                    todo(0) = j;//todo(1) = j // ?????
                } else {
                    todo(td1) = j;
                    td1++;
                }
            }
        } // 52 CONTINUE
        for( int z = 0; z < td1; z++ ) {
            j = todo(z);
            if( y(j) == -1 ) {// if( y(j) == 0 ) {
                goto M2;//M80;
            }
            ok(j) = true;
        } // 53 CONTINUE

        td2 = n;
        last = n + 1;

        // REPEAT UNTIL A FREE ROW HAS BEEN FOUND
        j0 = todo(td1);
        td1 = td1 - 1;

        todo(td) = j0;
        td = td - 1;

        i = y[j0];
        todo[td2] = j0;
        td2 = td - 1;

        t = first[i];

        while( kk[t] != j0 ) {
            t = t + 1;
        }
        h = cc_[t] - v(j0) - min_;
        for( t = t0; t < first[i + 1]; t++ ) {
            j = kk[t];
            if( !ok(j) ) {
                vj = cc_[t] - h - v(j);
                if( vj < d(j) ) {
                    d(j) = vj;
                    lab(j) = i;
                    if( std::abs( vj - min_ ) < resolution ) { // vj == min_
                        //if( std::abs( y(j) ) < resolution ) {
                        if( y(j) < 0 ) {
                            goto M1;
                        }
                        todo(td1) = j;
                        td1 = td1 + 1; // MY
                        ok(j) = true;
                    }
                }
            }
        }
        if( td1 = 0 ) {
            min_ = maxcost - 1;
            last = td2 + 1;
            for( j = 0; j < n; j++ ) {
                if( ( d(j) < min_ ) || ( std::abs( d(j) - min_ ) < resolution ) ) {
                    if( !ok(j) ) {
                        if( d(j) < min_ ) {
                            td1 = 0;
                            min_ = d(j);
                        }
                        todo[td1] = j;
                        td1 = td1 + 1; // MY
                    }
                }
            }
            for( h = 0; h < td1; h++ ) {
                j = todo(h);
                if( y(j) < 0 ) {
                    goto M1;
                }
                ok(j) = true;
            }
        }
        // UNTIL FALSE
M1:     for( k = last; k < n; k++ ) {
            j0 = todo(k);
            v(j0) = v(j0) + d(j0) - min_;
        }
M2:     do {
            i = lab(j);
            y(j) = i;
            k = j;
            j = x(i);
            x(i) = k;
        } while( i == i0 );
    }

    h = 0;
    for( i = 0; i < n; i++ ) {
        j = x(i);
        t = first[i];
        while( kk[t] != j ) {
            t++;
        }
        u(i) = cc_[t] - v(j);
        h = h + cc_[t];
    }

    lapcost = h;

    } // if( l != 0 )

}
*/

}
}
/// \}
