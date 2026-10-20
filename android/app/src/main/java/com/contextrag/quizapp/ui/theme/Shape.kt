package com.contextrag.quizapp.ui.theme

import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

// ===================================================
// Material Design 3 & M3 Expressive Shape Scale (10 Steps)
// ===================================================

object M3CornerRadius {
    val None: Dp = 0.dp
    val ExtraSmall: Dp = 4.dp
    val Small: Dp = 8.dp
    val Medium: Dp = 12.dp
    val Large: Dp = 16.dp
    /** Added in M3 Expressive */
    val LargeIncreased: Dp = 20.dp
    val ExtraLarge: Dp = 28.dp
    /** Added in M3 Expressive */
    val ExtraLargeIncreased: Dp = 32.dp
    /** Added in M3 Expressive */
    val ExtraExtraLarge: Dp = 48.dp
    /** Full pill corner radius */
    val Full: Dp = 24.dp
}

object M3Shapes {
    val CornerNone: Shape = RectangleShape
    val CornerExtraSmall = RoundedCornerShape(M3CornerRadius.ExtraSmall)
    val CornerSmall = RoundedCornerShape(M3CornerRadius.Small)
    val CornerMedium = RoundedCornerShape(M3CornerRadius.Medium)
    val CornerLarge = RoundedCornerShape(M3CornerRadius.Large)
    val CornerLargeIncreased = RoundedCornerShape(M3CornerRadius.LargeIncreased)
    val CornerExtraLarge = RoundedCornerShape(M3CornerRadius.ExtraLarge)
    val CornerExtraLargeIncreased = RoundedCornerShape(M3CornerRadius.ExtraLargeIncreased)
    val CornerExtraExtraLarge = RoundedCornerShape(M3CornerRadius.ExtraExtraLarge)
    val CornerFull: Shape = CircleShape

    /**
     * Optical roundness rule from M3 Expressive:
     * inner radius = outer radius - padding
     */
    fun nestedRadius(outerRadius: Dp, padding: Dp, minRadius: Dp = M3CornerRadius.ExtraSmall): Dp =
        (outerRadius - padding).coerceAtLeast(minRadius)

    fun nestedShape(outerRadius: Dp, padding: Dp, minRadius: Dp = M3CornerRadius.ExtraSmall): RoundedCornerShape =
        RoundedCornerShape(nestedRadius(outerRadius, padding, minRadius))
}

val Shapes = Shapes(
    extraSmall = M3Shapes.CornerExtraSmall,
    small = M3Shapes.CornerSmall,
    medium = M3Shapes.CornerMedium,
    large = M3Shapes.CornerLarge,
    extraLarge = M3Shapes.CornerExtraLarge
)
