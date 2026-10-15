//----------------------------------------------------------------------------------------------------------------------
///
/// \file       seqextr.hpp
/// \brief      Последовательный выбор экстремума
/// \addtogroup spm
/// \{
///

#pragma once

#include <armadillo>
#include <set>

#include "spm/common/compare.hpp"
#include "spm/lap/sparse.hpp"
#include "spm/lap/searchparam.hpp"

namespace SPM /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
///
/// \brief Процедура последовательного поиска экстремума по строкам/столбцам матрицы ценностей
/// \details Неоптимальная процедура, сумма назначений будет неоптимальна, поскольку строки/столбцы с найденным
/// экстремумом исключаются из анализа.
/// \param[in]  assigncost - матрица ценности
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  infValue   - большое положительное число
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
///
void SequentalExtremum( const arma::mat &assigncost, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost );

///
/// \brief Процедура последовательного поиска экстремума по строкам/столбцам матрицы ценностей
/// \details Неоптимальная процедура, сумма назначений будет неоптимальна, поскольку строки/столбцы с найденным
/// экстремумом исключаются из анализа.
/// \param[in]  assigncost - матрица ценности в сжатом COO формате
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  infValue   - большое положительное число
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
///
void SequentalExtremum( const Sparse::CMatrixCOO &assigncost, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost );

} // namespace LAP
} // namespace SPM
/// \}
