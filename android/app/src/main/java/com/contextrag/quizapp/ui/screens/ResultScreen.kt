package com.contextrag.quizapp.ui.screens

import androidx.activity.compose.BackHandler
import androidx.compose.animation.core.Animatable
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.EmojiEvents
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.MenuBook
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.ui.components.SectionHeader
import com.contextrag.quizapp.ui.components.expressiveBounceClick
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.M3ExpressiveMotion
import com.contextrag.quizapp.ui.theme.M3Shapes
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
import com.contextrag.quizapp.ui.theme.displayMediumEmphasized
import com.contextrag.quizapp.ui.theme.titleLargeEmphasized
import com.contextrag.quizapp.ui.theme.titleMediumEmphasized
import com.contextrag.quizapp.viewmodel.UiState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    state: UiState,
    onRetryQuiz: () -> Unit,
    onBackToChapters: () -> Unit,
    onBackToLibrary: () -> Unit
) {
    val result = state.evaluationResult

    // Donanım geri tuşu bölümlere döndürsün
    BackHandler {
        onBackToChapters()
    }

    if (result == null) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            contentAlignment = Alignment.Center
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = M3Shapes.CornerExtraLarge,
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                )
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Surface(
                        shape = M3Shapes.CornerLarge,
                        color = MaterialTheme.colorScheme.surfaceContainerHigh,
                        modifier = Modifier.size(64.dp)
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(
                                Icons.Default.EmojiEvents,
                                contentDescription = null,
                                modifier = Modifier.size(36.dp),
                                tint = MaterialTheme.colorScheme.outline
                            )
                        }
                    }
                    Text(
                        text = "Henüz Tamamlanan Quiz Yok",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Bir quiz çözüp bitirdiğinizde detaylı sonuçlarınız, doğru/yanlış analiziniz ve açıklamalar burada görünecektir.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Button(
                        onClick = onBackToChapters,
                        shape = M3Shapes.CornerFull,
                        modifier = Modifier.expressiveBounceClick(onClick = onBackToChapters)
                    ) {
                        Text("Bölümlere Git")
                    }
                }
            }
        }
        return
    }

    val isSuccess = result.scorePercentage >= 50f

    // M3 Expressive Spring Animated Score Counter
    val animatedScore = remember { Animatable(0f) }
    LaunchedEffect(result.scorePercentage) {
        animatedScore.animateTo(
            targetValue = result.scorePercentage,
            animationSpec = M3ExpressiveMotion.defaultSpatial()
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Sınav Değerlendirmesi",
                        style = MaterialTheme.typography.titleMediumEmphasized
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBackToChapters) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Bölümlere Dön")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(top = 10.dp, bottom = 48.dp)
        ) {
            // 1. Skor Özeti Kartı (M3 Expressive Celebratory Hero Card)
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = M3Shapes.CornerExtraLargeIncreased,
                    colors = CardDefaults.cardColors(
                        containerColor = if (isSuccess) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)
                        else MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.5f)
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Surface(
                            shape = M3Shapes.CornerFull,
                            color = if (isSuccess) CacheHitGreenContainer else MaterialTheme.colorScheme.errorContainer
                        ) {
                            Text(
                                text = result.resultBadge,
                                modifier = Modifier.padding(horizontal = 14.dp, vertical = 6.dp),
                                style = MaterialTheme.typography.titleMediumEmphasized,
                                color = if (isSuccess) OnCacheHitGreen else MaterialTheme.colorScheme.onErrorContainer
                            )
                        }

                        // Animated Percentage
                        Text(
                            text = "%${animatedScore.value.toInt()}",
                            style = MaterialTheme.typography.displayMediumEmphasized,
                            color = if (isSuccess) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.error
                        )

                        Text(
                            text = "${result.correctCount} / ${result.totalQuestions} Doğru",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface
                        )

                        // 3 Temel Aksiyon Butonu (M3 Expressive Bouncy CTAs)
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                FilledTonalButton(
                                    onClick = onRetryQuiz,
                                    modifier = Modifier
                                        .weight(1f)
                                        .expressiveBounceClick(onClick = onRetryQuiz),
                                    shape = M3Shapes.CornerFull
                                ) {
                                    Icon(Icons.Default.Replay, contentDescription = null, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Tekrar Çöz")
                                }

                                Button(
                                    onClick = onBackToChapters,
                                    modifier = Modifier
                                        .weight(1f)
                                        .expressiveBounceClick(onClick = onBackToChapters),
                                    shape = M3Shapes.CornerFull
                                ) {
                                    Text("Bölümlere Dön")
                                }
                            }

                            OutlinedButton(
                                onClick = onBackToLibrary,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .expressiveBounceClick(onClick = onBackToLibrary),
                                shape = M3Shapes.CornerFull
                            ) {
                                Icon(Icons.Default.MenuBook, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Kitaplığıma Dön")
                            }
                        }
                    }
                }
            }

            item {
                SectionHeader(
                    title = "Detaylı Soru İncelemesi",
                    badgeText = "${result.details.size} Soru"
                )
            }

            // 2. Soru İnceleme Kartları
            itemsIndexed(result.details) { idx, detail ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = M3Shapes.CornerLarge,
                    border = BorderStroke(
                        width = 1.dp,
                        color = if (detail.isCorrect) CacheHitGreen.copy(alpha = 0.6f) else MaterialTheme.colorScheme.error.copy(alpha = 0.5f)
                    ),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Soru ${idx + 1}",
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )

                            if (detail.isCorrect) {
                                Surface(
                                    shape = M3Shapes.CornerFull,
                                    color = CacheHitGreenContainer
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.CheckCircle,
                                            contentDescription = null,
                                            tint = CacheHitGreen,
                                            modifier = Modifier.size(14.dp)
                                        )
                                        Text(
                                            text = "Doğru",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = OnCacheHitGreen,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            } else {
                                Surface(
                                    shape = M3Shapes.CornerFull,
                                    color = MaterialTheme.colorScheme.errorContainer
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.Cancel,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.error,
                                            modifier = Modifier.size(14.dp)
                                        )
                                        Text(
                                            text = "Yanlış",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onErrorContainer,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }
                        }

                        Text(
                            text = detail.question,
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium
                        )

                        HorizontalDivider(
                            modifier = Modifier.padding(vertical = 4.dp),
                            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)
                        )

                        Text(
                            text = "Sizin Cevabınız: ${detail.userSelected}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = if (detail.isCorrect) CacheHitGreen else MaterialTheme.colorScheme.error,
                            fontWeight = FontWeight.SemiBold
                        )

                        if (!detail.isCorrect) {
                            Text(
                                text = "Doğru Cevap: ${detail.correctOption}",
                                style = MaterialTheme.typography.bodyMedium,
                                color = CacheHitGreen,
                                fontWeight = FontWeight.SemiBold
                            )
                        }

                        // Açıklama Kutusu
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            shape = M3Shapes.CornerSmall,
                            color = MaterialTheme.colorScheme.surfaceContainerHigh
                        ) {
                            Row(
                                modifier = Modifier.padding(12.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Icon(
                                    Icons.Default.Info,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = detail.explanation,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
