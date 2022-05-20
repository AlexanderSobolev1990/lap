//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap_jvc.cpp
/// \brief      Решение задачи о назначениях методом JVC (cтандартная линейная дискретная оптимизационная задача)
/// \date       18.05.22 - создан
/// \author     Соболев А.А.
///

#include <lap.h>

namespace LAP /// Решение задачи о назначениях
{
//----------------------------------------------------------------------------------------------------------------------
void CAssignmentProblemSolver::JVC( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol )
{
    // Если ищем максимум - умножим матрицу на -1
    arma::mat cost( dim, dim, arma::fill::zeros );
    if( sp == TSearchParam::Max ) { // Поиск минимума/максимума
        cost = -assigncost;
    } else {
        cost = assigncost;
    }
    bool unassignedfound;
    int i = 0, imin = 0, numfree = 0, prvnumfree = 0, f = 0, i0 = 0, k = 0, freerow = 0; // row
    int j = 0, j1 = 0, j2 = 0, endofpath = 0, last = 0, low = 0, up = 0;
    double dmin = 0.0, h = 0.0, umin = 0.0, usubmin = 0.0, v2 = 0.0; // cost

    arma::vec u = arma::vec( dim, arma::fill::zeros );//new double[dim];
    arma::vec v = arma::vec( dim, arma::fill::zeros );//new double[dim];
    arma::ivec free = arma::ivec( dim, arma::fill::zeros );//new int[dim];       // list of unassigned rows.
    arma::ivec collist = arma::ivec( dim, arma::fill::zeros );//new int[dim];    // list of columns to be scanned in various ways.
    arma::ivec matches = arma::ivec( dim, arma::fill::zeros );//new int[dim];    // counts how many times a row could be assigned.
    arma::vec d = arma::vec( dim, arma::fill::zeros );//new double[dim];       // 'cost-distance' in augmenting path calculation.
    arma::ivec pred = arma::ivec( dim, arma::fill::zeros );//new int[dim];       // row-predecessor of column in augmenting/alternating path.
    //int *rowsol = new int[dim];
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
        dmin = cost(0,j);
        imin = 0;
        for( i = 1; i < dim; i++ ) {
            if( cost(i,j) < dmin ) {
                dmin = cost(i,j);
                imin = i;
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
            rowsol(i) = j1;
            if( i == freerow ) {
                break;
            }
        }
    }

//    // calculate optimal cost.
//    //double lapcost = 0;
//    for( i = 0; i < dim; i++ ) {
//        j = rowsol(i);
//        u[i] = cost(i,j) - v(j);
//        //lapcost = lapcost + assigncost[( i * dim ) + j]; //lapcost = lapcost + assigncost[i][j];
//    }
}

}
