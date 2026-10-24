package com.contextrag.quizapp.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.AnimationVector1D
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.InteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.ContentDrawScope
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.node.DrawModifierNode
import androidx.compose.ui.node.ModifierNodeElement
import androidx.compose.ui.node.PointerInputModifierNode
import androidx.compose.ui.node.invalidateDraw
import androidx.compose.ui.platform.InspectorInfo
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.ui.theme.M3CornerRadius
import com.contextrag.quizapp.ui.theme.M3ExpressiveMotion
import com.contextrag.quizapp.ui.theme.M3Shapes
import com.contextrag.quizapp.ui.theme.headlineSmallEmphasized
import com.contextrag.quizapp.ui.theme.labelSmallEmphasized
import com.contextrag.quizapp.ui.theme.titleMediumEmphasized
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

// ==============================================================================
// 1. M3 Expressive Spring Bounce Click Modifiers
// Fast Spatial Spring (damping = 0.6f, stiffness = 800f) for physical tactile feedback
// ==============================================================================

fun Modifier.expressiveBounceClick(
    scaleDown: Float = 0.96f,
    enabled: Boolean = true,
    onClick: () -> Unit
): Modifier = this
    .expressivePressScale(
        interactionSource = null,
        scaleDown = scaleDown,
        enabled = enabled
    )
    .clickable(
        enabled = enabled,
        onClick = onClick
    )

fun Modifier.expressivePressScale(
    interactionSource: InteractionSource? = null,
    scaleDown: Float = 0.96f,
    enabled: Boolean = true
): Modifier = this then ExpressivePressScaleElement(
    interactionSource = interactionSource,
    scaleDown = scaleDown,
    enabled = enabled
)

private data class ExpressivePressScaleElement(
    val interactionSource: InteractionSource?,
    val scaleDown: Float,
    val enabled: Boolean
) : ModifierNodeElement<ExpressivePressScaleNode>() {
    override fun create(): ExpressivePressScaleNode =
        ExpressivePressScaleNode(scaleDown, enabled)

    override fun update(node: ExpressivePressScaleNode) {
        node.updateNode(scaleDown, enabled)
    }

    override fun InspectorInfo.inspectableProperties() {
        name = "expressivePressScale"
        properties["scaleDown"] = scaleDown
        properties["enabled"] = enabled
    }
}

private class ExpressivePressScaleNode(
    private var scaleDown: Float,
    private var enabled: Boolean
) : Modifier.Node(), PointerInputModifierNode, DrawModifierNode {
    private var scaleAnimatable: Animatable<Float, AnimationVector1D>? = null
    private var animJob: Job? = null
    private var isPointerPressed: Boolean = false

    fun updateNode(newScaleDown: Float, newEnabled: Boolean) {
        scaleDown = newScaleDown
        enabled = newEnabled
        if (!enabled && isPointerPressed) {
            isPointerPressed = false
            animateToScale(1f)
        }
    }

    override fun onPointerEvent(pointerEvent: PointerEvent, pass: PointerEventPass, bounds: IntSize) {
        if (pass != PointerEventPass.Main || !enabled) return
        when (pointerEvent.type) {
            PointerEventType.Press -> {
                if (!isPointerPressed && pointerEvent.changes.any { it.pressed && !it.isConsumed }) {
                    isPointerPressed = true
                    animateToScale(scaleDown)
                }
            }
            PointerEventType.Move -> {
                if (isPointerPressed && pointerEvent.changes.any { it.isConsumed || !it.pressed }) {
                    isPointerPressed = false
                    animateToScale(1f)
                }
            }
            PointerEventType.Release, PointerEventType.Exit -> {
                if (isPointerPressed && pointerEvent.changes.none { it.pressed }) {
                    isPointerPressed = false
                    animateToScale(1f)
                }
            }
        }
    }

    override fun onCancelPointerInput() {
        if (isPointerPressed) {
            isPointerPressed = false
            animateToScale(1f)
        }
    }

    override fun onDetach() {
        isPointerPressed = false
        animJob?.cancel()
        animJob = null
        scaleAnimatable = null
    }

    private fun animateToScale(target: Float) {
        animJob?.cancel()
        val anim = scaleAnimatable ?: Animatable(1f).also { scaleAnimatable = it }
        if (anim.value == target && !anim.isRunning) return
        animJob = coroutineScope.launch {
            anim.animateTo(
                targetValue = target,
                animationSpec = M3ExpressiveMotion.fastSpatial()
            ) {
                invalidateDraw()
            }
            invalidateDraw()
        }
    }

    override fun ContentDrawScope.draw() {
        val currentScale = scaleAnimatable?.value ?: 1f
        if (currentScale != 1f) {
            scale(scaleX = currentScale, scaleY = currentScale, pivot = center) {
                this@draw.drawContent()
            }
        } else {
            drawContent()
        }
    }
}

// ==============================================================================
// 2. MD3 Metric Card (From Personal Records)
// Displays statistics, cache statuses, and metrics with tactile bounce
// ==============================================================================

@Composable
fun MetricCard(
    title: String,
    value: String,
    badgeText: String = "",
    note: String = "",
    modifier: Modifier = Modifier,
    badgeBg: Color = MaterialTheme.colorScheme.secondaryContainer,
    badgeColor: Color = MaterialTheme.colorScheme.onSecondaryContainer,
    valueIcon: ImageVector? = null,
    valueIconTint: Color = MaterialTheme.colorScheme.primary,
    noteIcon: ImageVector? = null,
    noteIconTint: Color? = null,
    onClick: (() -> Unit)? = null
) {
    val cardModifier = if (onClick != null) {
        modifier.fillMaxWidth().expressiveBounceClick(onClick = onClick)
    } else {
        modifier.fillMaxWidth()
    }

    Card(
        modifier = cardModifier,
        shape = M3Shapes.CornerMedium,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.labelSmallEmphasized,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f, fill = false)
                )
                if (badgeText.isNotBlank()) {
                    Spacer(modifier = Modifier.width(6.dp))
                    Surface(
                        shape = M3Shapes.CornerFull,
                        color = badgeBg
                    ) {
                        Text(
                            text = badgeText,
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = badgeColor,
                            maxLines = 1,
                            softWrap = false
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                if (valueIcon != null) {
                    Icon(
                        imageVector = valueIcon,
                        contentDescription = null,
                        tint = valueIconTint,
                        modifier = Modifier.size(20.dp)
                    )
                }
                Text(
                    text = value,
                    style = MaterialTheme.typography.headlineSmallEmphasized,
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            if (note.isNotBlank()) {
                Spacer(modifier = Modifier.height(2.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    if (noteIcon != null) {
                        Icon(
                            imageVector = noteIcon,
                            contentDescription = null,
                            tint = noteIconTint ?: MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(12.dp)
                        )
                    }
                    Text(
                        text = note,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}

// ==============================================================================
// 3. Shimmer Loading Card
// Smooth continuous gradient animation for placeholder items
// ==============================================================================

@Composable
fun ShimmerCard(
    modifier: Modifier = Modifier,
    height: androidx.compose.ui.unit.Dp = 80.dp
) {
    val transition = rememberInfiniteTransition(label = "shimmer_transition")
    val translateX by transition.animateFloat(
        initialValue = -300f,
        targetValue = 600f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "shimmer_offset"
    )

    val shimmerColors = listOf(
        MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.5f),
        MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.9f),
        MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.5f)
    )

    val brush = Brush.linearGradient(
        colors = shimmerColors,
        start = Offset(translateX, 0f),
        end = Offset(translateX + 300f, 0f)
    )

    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(height)
            .clip(M3Shapes.CornerMedium)
            .background(brush)
    )
}

// ==============================================================================
// 4. Section Header with M3 Emphasized Typography
// ==============================================================================

@Composable
fun SectionHeader(
    title: String,
    modifier: Modifier = Modifier,
    badgeText: String? = null,
    badgeColor: Color = MaterialTheme.colorScheme.primaryContainer,
    badgeTextColor: Color = MaterialTheme.colorScheme.onPrimaryContainer
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMediumEmphasized,
            color = MaterialTheme.colorScheme.primary
        )
        if (!badgeText.isNullOrBlank()) {
            Surface(
                shape = M3Shapes.CornerFull,
                color = badgeColor
            ) {
                Text(
                    text = badgeText,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                    style = MaterialTheme.typography.labelSmallEmphasized,
                    color = badgeTextColor
                )
            }
        }
    }
}
