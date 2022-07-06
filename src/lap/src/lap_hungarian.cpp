//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap_hungarian.cpp
/// \brief      Решение задачи о назначениях венгерским методом (Hungarian, Munkres) (cтандартная линейная дискретная оптимизационная задача)
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
void hungarian_step_1( unsigned int &step, arma::mat &cost, const unsigned int &N )
{
    for( unsigned int r = 0; r < N; ++r ) {
        cost.row(r) -= arma::min( cost.row(r) );
    }
    step = 2;
}

/**
 * Note, that we use references for all function arguments.
 * As we have to switch between the steps of the algorithm
 * continously, we always must be able to determine which
 * step should be chosen next. Therefore we give a mutable
 * unsigned integer `step` as an argument to each step
 * function of the algorithm.
 *
 * Inside the function we can easily access a whole row by
 * Armadillo's `row()` method for matrices.
 * In the second step, we then search for a zero in the
 * modified cost matrix of step one.
 */
void hungarian_step_2( unsigned int &step, const arma::mat &cost, arma::umat &indM, arma::ivec &rcov,
    arma::ivec &ccov, const unsigned int &N )
{
    for( unsigned int r = 0; r < N; ++r ) {
        for( unsigned int c = 0; c < N; ++c ) {
            if( cost.at(r, c) == 0.0 && rcov.at(r) == 0 && ccov.at(c) == 0) {
                indM.at(r, c) = 1;
                rcov.at(r)    = 1;
                ccov.at(c)    = 1;
                break;                                              // Only take the first
                                                                    // zero in a row and column
            }
        }
    }
    /* for later reuse */
    rcov.fill(0);
    ccov.fill(0);
    step = 3;
}

/**
 * Only the first zero in a row is taken. Then, the indicator
 * matrix `indM` indicates this zero by setting the corresponding
 * element at `(r, c)` to 1. A unique zero - the only or first one in
 * a column and row - is called _starred zero_. In `step 2` we find
 * such a _starred zero_.
 *
 * Note, that we use here Armadillo's element access via the
 * method `at()`, which makes no bound checks and improves performance.
 *
 * _Note Bene: This code is thoroughly debugged - never do this for fresh written
 * code!_
 *
 * In `step 3` we cover each column with a _starred zero_. If already
 * `N` columns are covered all _starred zeros_ describe a complete
 * assignment - so, go to `step 7` and finish. Otherwise go to
 * `step 4`.
 */
void hungarian_step_3( unsigned int &step, const arma::umat &indM, arma::ivec &ccov, const unsigned int &N )
{
    unsigned int colcount = 0;
    for( unsigned int r = 0; r < N; ++r ) {
        for( unsigned int c = 0; c < N; ++c ) {
            if( indM.at(r, c) == 1 ) {
                ccov.at(c) = 1;
            }
        }
    }
    for( unsigned int c = 0; c < N; ++c ) {
        if( ccov.at(c) == 1 ) {
            ++colcount;
        }
    }
    if( colcount == N ) {
        step = 7;
    } else {
        step = 4;
    }
}

/**
 * We cover a column by looking for 1s in the indicator
 * matrix `indM` (See `step 2` for assuring that these are
 * indeed only _starred zeros_).
 *
 * `Step 4` finds _noncovered zeros_ and _primes_ them. If there
 * are zeros in a row and none of them is _starred_, _prime_
 * them. For this task we program a helper function to keep
 * the code more readable and reusable. The helper function
 * searches for _noncovered zeros_.
 */
void hungarian_find_noncovered_zero( int &row, int &col, const arma::mat &cost, const arma::ivec &rcov,
    const arma::ivec &ccov, const unsigned int &N )
{
    unsigned int r = 0;
    unsigned int c;
    bool done = false;
    row = -1;
    col = -1;
    while( !done ) {
        c = 0;
        while( true ) {
            if( cost.at(r, c) == 0.0 && rcov.at(r) == 0 && ccov.at(c) == 0 ) {
                row = r;
                col = c;
                done = true;
            }
            ++c;
            if( c == N || done ) {
                break;
            }
        }
        ++r;
        if( r == N ) {
            done = true;
        }
    }
}

/**
 * We can detect _noncovered zeros_ by checking if the cost matrix
 * contains at row r and column c a zero and row and column
 * are not covered yet, i.e. `rcov(r) == 0`, `ccov(c) == 0`.
 * This loop breaks, if we have found our first _uncovered zero_  or
 * no _uncovered zero_ at all.
 *
 * In `step 4`, if no _uncovered zero_ is found we go to `step 6`. If
 * instead an _uncovered zero_ has been found, we set the indicator
 * matrix at its position to 2. We then have to search for a _starred
 * zero_ in the row with the _uncovered zero_, _uncover_ the column with
 * the _starred zero_ and _cover_ the row with the _starred zero_. To
 * indicate a _starred zero_ in a row and to find it we create again
 * two helper functions.
 */
bool hungarian_star_in_row( int &row, const arma::umat &indM, const unsigned int &N )
{
    bool tmp = false;
    for( unsigned int c = 0; c < N; ++c ) {
        if( indM.at(row, c) == 1 ) {
            tmp = true;
            break;
        }
    }
    return tmp;
}

void hungarian_find_star_in_row( const int &row, int &col, const arma::umat &indM, const unsigned int &N )
{
    col = -1;
    for( unsigned int c = 0; c < N; ++c ) {
        if( indM.at(row, c) == 1 ) {
            col = c;
        }
    }
}

/**
 * We know that _starred zeros_ are indicated by the indicator
 * matrix containing an element equal to 1.
 * Now, `step 4`.
 */
void hungarian_step_4( unsigned int &step, const arma::mat &cost, arma::umat &indM, arma::ivec &rcov, arma::ivec &ccov,
    int &rpath_0, int &cpath_0, const unsigned int &N )
{
    int row = -1;
    int col = -1;
    bool done = false;
    while( !done ) {
        hungarian_find_noncovered_zero( row, col, cost, rcov, ccov, N );
        if( row == -1 ) {
            done = true;
            step = 6;
        } else {
            /* uncovered zero */
            indM( row, col ) = 2;
            if( hungarian_star_in_row( row, indM, N ) ) {
                hungarian_find_star_in_row( row, col, indM, N );
                /* Cover the row with the starred zero
                 * and uncover the column with the starred
                 * zero.
                 */
                rcov.at(row) = 1;
                ccov.at(col) = 0;
            } else {
                /* No starred zero in row with
                 * uncovered zero
                 */
                done = true;
                step = 5;
                rpath_0 = row;
                cpath_0 = col;
            }
        }
    }
}

/**
 * Notice the `rpath_0` and `cpath_0` variables. These integer
 * variables store the first _vertex_ for an _augmenting path_ in `step 5`.
 * If zeros could be _primed_ we go further to `step 5`.
 *
 * `Step 5` constructs a path beginning at an _uncovered primed
 * zero_ (this is actually graph theory - alternating and augmenting
 * paths) and alternating between _starred_ and _primed zeros_.
 * This path is continued until a _primed zero_ with no _starred
 * zero_ in its column is found. Then, all _starred zeros_ in
 * this path are _unstarred_ and all _primed zeros_ are _starred_.
 * All _primes_ in the indicator matrix are erased and all rows
 * are _uncovered_. Then return to `step 3` to _cover_ again columns.
 *
 * `Step 5` needs several helper functions. First, we need
 * a function to find _starred zeros_ in columns.
 */
void hungarian_find_star_in_col( const int &col, int &row, const arma::umat &indM, const unsigned int &N )
{
    row = -1;
    for( unsigned int r = 0; r < N; ++r ) {
        if( indM.at(r, col) == 1 ) {
            row = r;
        }
    }
}

/**
 * Then we need a function to find a _primed zero_ in a row.
 * Note, that these tasks are easily performed by searching the
 * indicator matrix `indM`.
 */
void hungarian_find_prime_in_row( const int &row, int &col, const arma::umat &indM, const unsigned int &N )
{
    for( unsigned int c = 0; c < N; ++c ) {
        if( indM.at(row, c) == 2 ) {
            col = c;
        }
    }
}

/**
 * In addition we need a function to augment the path, one to
 * clear the _covers_ from rows and one to erase the _primed zeros_
 * from the indicator matrix `indM`.
 */
void hungarian_augment_path( const int &path_count, arma::umat &indM, const arma::imat &path )
{
//    for (unsigned int p = 0; p < path_count; ++p) {
    for( int p = 0; p < path_count; ++p ) {
        if( indM.at( path(p, 0), path(p, 1) ) == 1) {
            indM.at( path(p, 0), path(p, 1) ) = 0;
        } else {
            indM.at( path(p, 0), path(p, 1) ) = 1;
        }
    }
}

void hungarian_clear_covers( arma::ivec &rcov, arma::ivec &ccov )
{
    rcov.fill(0);
    ccov.fill(0);
}

void hungarian_erase_primes( arma::umat &indM, const unsigned int &N )
{
    for( unsigned int r = 0; r < N; ++r ) {
        for( unsigned int c = 0; c < N; ++c ) {
            if( indM.at(r, c) == 2 ) {
                indM.at(r, c) = 0;
            }
        }
    }
}

/**
 * The function to augment the path gets an integer matrix `path`
 * of dimension 2 * N x 2. In it all vertices between rows and columns
 * are stored row-wise.
 * Now, we can set the complete `step 5`:
 */
void hungarian_step_5( unsigned int &step, arma::umat &indM, arma::ivec &rcov, arma::ivec &ccov, arma::imat &path,
    int &rpath_0, int &cpath_0, const unsigned int &N )
{
    bool done = false;
    int row = -1;
    int col = -1;
    unsigned int path_count = 1;
    path.at(path_count - 1, 0) = rpath_0;
    path.at(path_count - 1, 1) = cpath_0;
    while( !done ) {
        hungarian_find_star_in_col( path.at(path_count - 1, 1), row, indM, N );
        if( row > -1 ) {
            /* Starred zero in row 'row' */
            ++path_count;
            path.at(path_count - 1, 0) = row;
            path.at(path_count - 1, 1) = path.at(path_count - 2, 1);
        } else {
            done = true;
        }
        if( !done ) {
            /* If there is a starred zero find a primed
             * zero in this row; write index to 'col' */
            hungarian_find_prime_in_row( path.at(path_count - 1, 0), col, indM, N);
            ++path_count;
            path.at(path_count - 1, 0) = path.at(path_count - 2, 0);
            path.at(path_count - 1, 1) = col;
        }
    }
    hungarian_augment_path( path_count, indM, path );
    hungarian_clear_covers( rcov, ccov );
    hungarian_erase_primes( indM, N );
    step = 3;
}

/**
 * Recall, if `step 4` was successfull in uncovering all columns
 * and covering all rows with a primed zero, it then calls
 * `step 6`.
 * `Step 6` takes the cover vectors `rcov` and `ccov` and looks
 * in the uncovered region of the cost matrix for the smallest
 * value. It then subtracts this value from each element in an
 * _uncovered column_ and adds it to each element in a _covered row_.
 * After this transformation, the algorithm starts again at `step 4`.
 * Our last helper function searches for the smallest value in
 * the uncovered region of the cost matrix.
 */
void hungarian_find_smallest( double &minval, const arma::mat &cost, const arma::ivec &rcov, const arma::ivec &ccov,
    const unsigned int &N )
{
    for( unsigned int r = 0; r < N; ++r ) {
        for( unsigned int c = 0; c < N; ++c ) {
            if( rcov.at(r) == 0 && ccov.at(c) == 0 ) {
                if( minval > cost.at(r, c) ) {
                    minval = cost.at(r, c);
                }
            }
        }
    }
}

/**
 * `Step 6` looks as follows:
 */
void hungarian_step_6( unsigned int &step, arma::mat &cost, const arma::ivec &rcov, const arma::ivec &ccov,
    const unsigned int &N )
{
    double minval = std::numeric_limits<double>::max();// DBL_MAX;
    hungarian_find_smallest( minval, cost, rcov, ccov, N );
    for( unsigned int r = 0; r < N; ++r ) {
        for( unsigned int c = 0; c < N; ++c ) {
            if( rcov.at(r) == 1 ) {
                cost.at(r, c) += minval;
            }
            if( ccov.at(c) == 0 ) {
                cost.at(r, c) -= minval;
            }
        }
    }
    step = 4;
}

/**
 * At last, we must create a function that enables us to
 * jump around the different steps of the algorithm.
 * The following code shows the main function of
 * the algorithm. It defines also the important variables
 * to be passed to the different steps.
 */
void Hungarian( const arma::mat &assigncost, int dim, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost )
{
    const unsigned int N = assigncost.n_rows;
    unsigned int step = 1;
    int cpath_0 = 0;
    int rpath_0 = 0;

    // Если ищем максимум - умножим матрицу на -1
    arma::mat cost( dim, dim, arma::fill::zeros );
    if( sp == TSearchParam::SP_Max ) { // Поиск минимума/максимума
        cost = -assigncost;
    } else {
        cost = assigncost;
    }

    arma::umat indM(N, N);
    arma::ivec rcov(N);
    arma::ivec ccov(N);
    arma::imat path(2 * N, 2);

    indM = arma::zeros<arma::umat>(N, N);
    bool done = false;
    while( !done ) {
        switch( step ) {
            case 1:
                hungarian_step_1( step, cost, N );
                break;
            case 2:
                hungarian_step_2( step, cost, indM, rcov, ccov, N );
                break;
            case 3:
                hungarian_step_3( step, indM, ccov, N );
                break;
            case 4:
                hungarian_step_4( step, cost, indM, rcov, ccov, rpath_0, cpath_0, N );
                break;
            case 5:
                hungarian_step_5( step, indM, rcov, ccov, path, rpath_0, cpath_0, N );
                break;
            case 6:
                hungarian_step_6( step, cost, rcov, ccov, N );
                break;
            case 7:
                done = true;
                break;
            default:
                assert( false );
        }
    }

    // Вывод результата
    lapcost = 0.0;
    for( int i = 0; i < dim; i++ ) {
        for( int j = 0; j < dim; j++ ) {
            if( indM(i, j) > 0 ) {
                rowsol[i] = j;
                double element_i_j = assigncost(i,j);
                if( !SPML::Compare::AreEqualAbs( element_i_j, infValue, resolution ) ) {
                    lapcost += element_i_j;
                }
            }
        }
    }
    return;
}

} // end namespace LAP
} // end namespace SPML
/// \}
