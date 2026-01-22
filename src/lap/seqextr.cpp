//----------------------------------------------------------------------------------------------------------------------
///
/// \file       seqextr.cpp
/// \brief      Последовательный выбор экстремума
/// \date       24.06.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <seqextr.hpp>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{

void SequentalExtremum( const arma::mat &assigncost, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost )
{
    size_t cols = assigncost.n_cols;
    size_t rows = assigncost.n_rows;
    arma::mat cost( rows, cols, arma::fill::zeros );
    if( sp == TSearchParam::SP_Max ) { // Поиск минимума/максимума
        cost = -assigncost;
    } else {
        cost = assigncost;
    }

    rowsol.zeros();
    rowsol -= 1;

    // Процедура всегда ищет минимум!

    // Вектора выкинутых индексов
    std::vector<int> cols_pulled;
    std::vector<int> rows_pulled;
    cols_pulled.reserve( cols );
    rows_pulled.reserve( rows );

    while( rows_pulled.size() < rows ) {

        double min_val = infValue;
        int min_row = INT32_MAX;
        int min_col = INT32_MAX;
        bool min_found = false;

        for( size_t row = 0; row < rows; row++ ) {
            if( std::find( rows_pulled.begin(), rows_pulled.end(), row ) != rows_pulled.end() ) {
                continue; // индекс row был выкинут
            }
            for( size_t col = 0; col < cols; col++ ) {
                if( std::find( cols_pulled.begin(), cols_pulled.end(), col ) != cols_pulled.end() ) {
                    continue; // индекс col был выкинут
                }
                if( ( min_val - cost(row,col) ) > resolution ) { // if( cost(i,j) < min_val ) {
                    min_val = cost(row,col);
                    min_row = row;
                    min_col = col;
                    min_found = true;
                }
            }
        }
        if( min_found ) {
            rows_pulled.push_back( min_row );
            cols_pulled.push_back( min_col );

            rowsol( min_row ) = min_col; // Решение!
        }
    }

    // calculate lapcost.
    lapcost = 0.0;
    for( size_t i = 0; i < rows; i++ ) {
        int j = rowsol(i);
        double element_i_j = assigncost(i,j);
        if( !SPML::Compare::AreEqualAbs( element_i_j, infValue, resolution ) ) {
            lapcost += element_i_j;
        }
    }
    return;
}

void SequentalExtremum( const Sparse::CMatrixCOO &assigncost, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost )
{
    rowsol.zeros();
    rowsol -= 1;

    // Нахождение числа строк/столбцов через std::set
    const std::size_t n_elems = assigncost.coo_val.size();
    std::set<int> pulled; // чтобы не морочиться на уникальность вхождений

    std::vector<double> cost = assigncost.coo_val;
    if( sp == TSearchParam::SP_Max ) {
        std::transform( cost.begin(), cost.end(), cost.begin(),
            std::bind( std::multiplies<double>(), std::placeholders::_1, -1.0 ) ); // Умножим на -1 для поиска максимума
    }

    // Процедура всегда ищет минимум!
    while( pulled.size() < n_elems ) {

        double min_val = infValue;
        int min_row = INT32_MAX;
        int min_col = INT32_MAX;
        bool min_found = false;

        for( size_t n = 0; n < n_elems; n++ ) {
            if( std::find( pulled.begin(), pulled.end(), n ) != pulled.end() ) {
                continue; // индексы выкинуты
            }
            if( ( min_val - cost[n] ) > resolution ) { // if( cost(n) < min_val ) {
                min_val = cost[n];
                min_row = assigncost.coo_row[n];
                min_col = assigncost.coo_col[n];
                min_found = true;
            }
        }
        if( min_found ) {
            for( size_t n = 0; n < n_elems; n++ ) {
                if( ( assigncost.coo_row[n] == min_row ) ||
                    ( assigncost.coo_col[n] == min_col ) )
                {
                    pulled.insert( n );
                }
            }
            rowsol( min_row ) = min_col; // Решение!
        }
    }
    // calculate lapcost.


    // Нахождение числа строк/столбцов через std::set
//    size_t cols = std::set<int>( ( assigncost.coo_col ).begin(), ( assigncost.coo_col ).end() ).size();
//    size_t rows = std::set<int>( ( assigncost.coo_row ).begin(), ( assigncost.coo_row ).end() ).size();
    lapcost = 0.0;
    for( size_t i = 0; i < rowsol.n_elem; i++ ) {
        if( i < 0 ) {
            continue;
        }
        int j = rowsol(i);
        if( j < 0 ) {
            continue;
        }
        for( size_t n = 0; n < n_elems; n++ ) {
            if( ( assigncost.coo_row[n] == i ) &&
                ( assigncost.coo_col[n] == j ) )
            {
                if( !SPML::Compare::AreEqualAbs( assigncost.coo_val[n], infValue, resolution ) ) {
                    lapcost += assigncost.coo_val[n];
//                    break;
                }
            }
        }
    }
    return;
}

} // end namespace LAP
} // end namespace SPML
/// \}
