package com.contextrag.quizapp.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.animation.core.SpringSpec
import androidx.compose.animation.core.spring
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = md_theme_light_primary,
    onPrimary = md_theme_light_onPrimary,
    primaryContainer = md_theme_light_primaryContainer,
    onPrimaryContainer = md_theme_light_onPrimaryContainer,
    secondary = md_theme_light_secondary,
    onSecondary = md_theme_light_onSecondary,
    secondaryContainer = md_theme_light_secondaryContainer,
    onSecondaryContainer = md_theme_light_onSecondaryContainer,
    tertiary = md_theme_light_tertiary,
    onTertiary = md_theme_light_onTertiary,
    tertiaryContainer = md_theme_light_tertiaryContainer,
    onTertiaryContainer = md_theme_light_onTertiaryContainer,
    error = md_theme_light_error,
    errorContainer = md_theme_light_errorContainer,
    onError = md_theme_light_onError,
    onErrorContainer = md_theme_light_onErrorContainer,
    background = md_theme_light_background,
    onBackground = md_theme_light_onBackground,
    surface = md_theme_light_surface,
    onSurface = md_theme_light_onSurface,
    surfaceVariant = md_theme_light_surfaceVariant,
    onSurfaceVariant = md_theme_light_onSurfaceVariant,
    outline = md_theme_light_outline,
    outlineVariant = md_theme_light_outlineVariant,
    inverseOnSurface = md_theme_light_inverseOnSurface,
    inverseSurface = md_theme_light_inverseSurface,
    inversePrimary = md_theme_light_inversePrimary,
    surfaceContainerLowest = md_theme_light_surfaceContainerLowest,
    surfaceContainerLow = md_theme_light_surfaceContainerLow,
    surfaceContainer = md_theme_light_surfaceContainer,
    surfaceContainerHigh = md_theme_light_surfaceContainerHigh,
    surfaceContainerHighest = md_theme_light_surfaceContainerHighest
)

private val DarkColorScheme = darkColorScheme(
    primary = md_theme_dark_primary,
    onPrimary = md_theme_dark_onPrimary,
    primaryContainer = md_theme_dark_primaryContainer,
    onPrimaryContainer = md_theme_dark_onPrimaryContainer,
    secondary = md_theme_dark_secondary,
    onSecondary = md_theme_dark_onSecondary,
    secondaryContainer = md_theme_dark_secondaryContainer,
    onSecondaryContainer = md_theme_dark_onSecondaryContainer,
    tertiary = md_theme_dark_tertiary,
    onTertiary = md_theme_dark_onTertiary,
    tertiaryContainer = md_theme_dark_tertiaryContainer,
    onTertiaryContainer = md_theme_dark_onTertiaryContainer,
    error = md_theme_dark_error,
    errorContainer = md_theme_dark_errorContainer,
    onError = md_theme_dark_onError,
    onErrorContainer = md_theme_dark_onErrorContainer,
    background = md_theme_dark_background,
    onBackground = md_theme_dark_onBackground,
    surface = md_theme_dark_surface,
    onSurface = md_theme_dark_onSurface,
    surfaceVariant = md_theme_dark_surfaceVariant,
    onSurfaceVariant = md_theme_dark_onSurfaceVariant,
    outline = md_theme_dark_outline,
    outlineVariant = md_theme_dark_outlineVariant,
    inverseOnSurface = md_theme_dark_inverseOnSurface,
    inverseSurface = md_theme_dark_inverseSurface,
    inversePrimary = md_theme_dark_inversePrimary,
    surfaceContainerLowest = md_theme_dark_surfaceContainerLowest,
    surfaceContainerLow = md_theme_dark_surfaceContainerLow,
    surfaceContainer = md_theme_dark_surfaceContainer,
    surfaceContainerHigh = md_theme_dark_surfaceContainerHigh,
    surfaceContainerHighest = md_theme_dark_surfaceContainerHighest
)

// ==========================================
// M3 Expressive Motion Physics Tokens
// ==========================================

object M3ExpressiveMotion {
    // Spatial = position, size, scale, translation (overshoots and bounces)
    const val EXPRESSIVE_DEFAULT_SPATIAL_DAMPING = 0.8f
    const val EXPRESSIVE_DEFAULT_SPATIAL_STIFFNESS = 380f

    const val EXPRESSIVE_FAST_SPATIAL_DAMPING = 0.6f
    const val EXPRESSIVE_FAST_SPATIAL_STIFFNESS = 800f

    const val EXPRESSIVE_SLOW_SPATIAL_DAMPING = 0.8f
    const val EXPRESSIVE_SLOW_SPATIAL_STIFFNESS = 200f

    // Effects = opacity, color (damping 1.0, never overshoots)
    const val EFFECTS_DAMPING = 1.0f
    const val DEFAULT_EFFECTS_STIFFNESS = 1600f
    const val FAST_EFFECTS_STIFFNESS = 3800f
    const val SLOW_EFFECTS_STIFFNESS = 800f

    fun <T> defaultSpatial(visibilityThreshold: T? = null): SpringSpec<T> =
        spring(
            dampingRatio = EXPRESSIVE_DEFAULT_SPATIAL_DAMPING,
            stiffness = EXPRESSIVE_DEFAULT_SPATIAL_STIFFNESS,
            visibilityThreshold = visibilityThreshold
        )

    fun <T> fastSpatial(visibilityThreshold: T? = null): SpringSpec<T> =
        spring(
            dampingRatio = EXPRESSIVE_FAST_SPATIAL_DAMPING,
            stiffness = EXPRESSIVE_FAST_SPATIAL_STIFFNESS,
            visibilityThreshold = visibilityThreshold
        )

    fun <T> slowSpatial(visibilityThreshold: T? = null): SpringSpec<T> =
        spring(
            dampingRatio = EXPRESSIVE_SLOW_SPATIAL_DAMPING,
            stiffness = EXPRESSIVE_SLOW_SPATIAL_STIFFNESS,
            visibilityThreshold = visibilityThreshold
        )

    fun <T> defaultEffects(visibilityThreshold: T? = null): SpringSpec<T> =
        spring(
            dampingRatio = EFFECTS_DAMPING,
            stiffness = DEFAULT_EFFECTS_STIFFNESS,
            visibilityThreshold = visibilityThreshold
        )

    fun <T> fastEffects(visibilityThreshold: T? = null): SpringSpec<T> =
        spring(
            dampingRatio = EFFECTS_DAMPING,
            stiffness = FAST_EFFECTS_STIFFNESS,
            visibilityThreshold = visibilityThreshold
        )

    fun <T> slowEffects(visibilityThreshold: T? = null): SpringSpec<T> =
        spring(
            dampingRatio = EFFECTS_DAMPING,
            stiffness = SLOW_EFFECTS_STIFFNESS,
            visibilityThreshold = visibilityThreshold
        )
}

@Composable
fun ContextRagTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.surface.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        shapes = Shapes,
        content = content
    )
}
