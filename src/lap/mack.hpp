//----------------------------------------------------------------------------------------------------------------------
///
/// \file       mack.hpp
/// \brief      Решение задачи о назначениях методом Мака для плотных матриц
/// \date       07.02.23 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_MACK_HPP_
#define SPML_MACK_HPP_

#include <armadillo>

#include <compare.hpp>
#include <searchparam.hpp>

namespace SPML /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
///
/// \brief Метод Мака решения задачи о назначениях
/// \details Источник: Банди Б. Основы линейного программирования: Пер. с англ. - М.:Радио м связь, 1989, стр 113-123
/// \param[in] assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in] dim - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in] sp - критерий поиска (минимум/максимум)
/// \param[in] infValue - большое положительное число
/// \param[in] resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
///
void Mack( const arma::mat &assigncost, int dim, TSearchParam sp, double infValue, double resolution, arma::ivec &rowsol,
    double &lapcost );

} // namespace LAP
} // namespace SPML
#endif
/// \}
