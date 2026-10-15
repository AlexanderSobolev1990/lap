//----------------------------------------------------------------------------------------------------------------------
///
/// \file       jvcdense.hpp
/// \brief      Решение задачи о назначениях методом JVC для плотных матриц
/// \addtogroup spm
/// \{
///

#pragma once

#include <armadillo>

#include "spm/common/compare.hpp"
#include "spm/lap/searchparam.hpp"

namespace SPM /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// плотных матриц
/// \details Источники:
/// 1)  "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
///     Assignment Problems," Computing 38, 325-340, 1987 by
///     R. Jonker and A. Volgenant, University of Amsterdam.
/// 2)  https://github.com/yongyanghz/LAPJV-algorithm-c
/// 3)  https://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0
/// \remarks - Оригинальный код R. Jonker and A. Volgenant [1] для целых чисел адаптирован под вещественные
///          - Метод подразделен на 4 процедуры в соответствии с модификацией Castanon:
///            * COLUMN REDUCTION
///            * REDUCTION TRANSFER
///            * AUGMENTING ROW REDUCTION - аукцион
///            * AUGMENT SOLUTION FOR EACH FREE ROW на основе алгоритма Dijkstra
///          - Правки из [3] касающиеся точности сравнения вещественных чисел
/// \param[in] assigncost - конечная матрица стоимости [dim,nc], где nc >= dim
/// \param[in] dim - число строк матрицы и размерность результата rowsol
/// \param[in] sp - критерий поиска (минимум/максимум)
/// \param[in] infValue - положительный конечный параметр совместимости; не ограничивает стоимость
/// \param[in] resolution - конечный неотрицательный допуск для сравнения вещественных чисел
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost - сумма назначенных элементов матрицы ценности assigncost
/// \details Каждая строка получает отдельный столбец; в горизонтальной матрице лишние столбцы остаются свободными.
/// Все конечные элементы, включая равные infValue, являются допустимыми стоимостями и входят в lapcost.
/// Фиктивные строки и столбцы автоматически не добавляются. Выходной rowsol получает размер dim.
/// Для нуля строк возвращаются пустое назначение и нулевая стоимость. При ошибке входа выходы не меняются.
/// \throw std::invalid_argument при несовпадении dim с числом строк, вертикальной матрице,
/// неконечных стоимостях или некорректных параметрах; std::length_error при слишком большом числе столбцов
///
void JVCdense( const arma::mat &assigncost, int dim, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost );

} // namespace LAP
} // namespace SPM
/// \}
