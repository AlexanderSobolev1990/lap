//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap_seqextr.cpp
/// \brief      Последовательный выбор экстремума на матрице
/// \date       24.06.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <lap.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{

void SequentalExtremum( const arma::mat &assigncost, TSearchParam sp, double maxcost, double resolution,
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

        double min_val = maxcost;
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
        if( !SPML::Compare::AreEqualAbs( element_i_j, maxcost, resolution ) ) {
            lapcost += element_i_j;
        }
    }
    return;
}

void SequentalExtremum( const Sparse::CMatrixCOO &assigncost, TSearchParam sp, double maxcost, double resolution,
    arma::ivec &rowsol, double &lapcost )
{
    // Нахождение числа строк/столбцов через std::set
//    size_t cols = std::set<int>( ( assigncost.coo_col ).begin(), ( assigncost.coo_col ).end() ).size();
//    size_t rows = std::set<int>( ( assigncost.coo_row ).begin(), ( assigncost.coo_row ).end() ).size();
    const std::size_t n_elems = assigncost.coo_val.size();

    std::vector<int> pulled;
//    pulled.resize( n_elems, false ); // Сброшено в false

    std::vector<double> cost = assigncost.coo_val;
    if( sp == TSearchParam::SP_Max ) {
        std::transform( cost.begin(), cost.end(), cost.begin(),
            std::bind( std::multiplies<double>(), std::placeholders::_1, -1.0 ) ); // Умножим на -1 для поиска максимума
    }

    // Процедура всегда ищет минимум!

//    // Вектора выкинутых индексов
//    std::vector<int> cols_pulled;
//    std::vector<int> rows_pulled;
//    cols_pulled.reserve( cols );
//    rows_pulled.reserve( rows );

    while( pulled.size() < n_elems ) {

        double min_val = maxcost;
        int min_row = INT32_MAX;
        int min_col = INT32_MAX;
//        int min_pos = INT32_MAX;
        bool min_found = false;

        for( size_t n = 0; n < n_elems; n++ ) {
//            if( ( std::find( rows_pulled.begin(), rows_pulled.end(), assigncost.coo_row[n] ) != rows_pulled.end() ) ||
//                ( std::find( cols_pulled.begin(), cols_pulled.end(), assigncost.coo_col[n] ) != cols_pulled.end() ) )
            if( std::find( pulled.begin(), pulled.end(), n ) != pulled.end() ) {
                continue; // индексы выкинуты
            }
            if( ( min_val - cost[n] ) > resolution ) { // if( cost(n) < min_val ) {
                min_val = cost[n];
                min_row = assigncost.coo_row[n];
                min_col = assigncost.coo_col[n];
//                min_pos = n;
                min_found = true;
            }
        }
        if( min_found ) {
            for( size_t n = 0; n < n_elems; n++ ) {
                if( ( assigncost.coo_row[n] == min_row ) ||
                    ( assigncost.coo_col[n] == min_col ) )
                {
                    pulled.push_back( n );
                }
            }
//            rows_pulled.push_back( min_row );
//            cols_pulled.push_back( min_col );

            rowsol( min_row ) = min_col; // Решение!
        }

    }
}

} // end namespace LAP
} // end namespace SPML
/// \}
