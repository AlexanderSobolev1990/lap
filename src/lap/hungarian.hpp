//----------------------------------------------------------------------------------------------------------------------
///
/// \file       hungarian.hpp
/// \brief      Решение задачи о назначениях венгерским методом (Hungarian, Munkres)
/// \date       07.02.23 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_HUNGARIAN_HPP_
#define SPML_HUNGARIAN_HPP_

#include <armadillo>
#include <cassert>

#include <compare.hpp>
#include <searchparam.hpp>

namespace SPML /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
///
/// \brief Венгерский метод решения задачи о назначениях (Метод Мункреса)
/// \details Источник: https://github.com/RcppCore/rcpp-gallery/blob/gh-pages/src/2013-09-24-minimal-assignment.cpp
/// \param[in] assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in] dim - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in] sp - критерий поиска (минимум/максимум)
/// \param[in] infValue - большое положительное число
/// \param[in] resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost - сумма назначенных элементов матрицы ценности assigncost
///
void Hungarian( const arma::mat &assigncost, int dim, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost );

} // namespace LAP
} // namespace SPML
#endif
/// \}
