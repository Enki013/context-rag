package com.contextrag.quizapp.ui.screens

import androidx.activity.compose.BackHandler
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
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
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.extraLarge
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Default.EmojiEvents,
                        contentDescription = null,
                        modifier = Modifier.size(56.dp),
                        tint = MaterialTheme.colorScheme.outline
                    )
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
                    Button(onClick = onBackToChapters, shape = CircleShape) {
                        Text("Bölümlere Git")
                    }
                }
            }
        }
        return
    }

    val isSuccess = result.scorePercentage >= 50f

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Sınav Değerlendirmesi",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
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
            contentPadding = PaddingValues(top = 8.dp, bottom = 48.dp)
        ) {
            // Skor Özeti Kartı (M3 ElevatedCard)
            item {
                ElevatedCard(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.extraLarge,
                    colors = CardDefaults.elevatedCardColors(
                        containerColor = if (isSuccess) MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = result.resultBadge,
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "%${result.scorePercentage.toInt()}",
                            style = MaterialTheme.typography.displayMedium,
                            fontWeight = FontWeight.ExtraBold,
                            color = if (isSuccess) MaterialTheme.colorScheme.onPrimaryContainer
                            else MaterialTheme.colorScheme.onErrorContainer
                        )

                        Text(
                            text = "${result.correctCount} / ${result.totalQuestions} Doğru",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )

                        // 3 Temel Aksiyon Butonu
                        Column(
                            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                FilledTonalButton(
                                    onClick = onRetryQuiz,
                                    modifier = Modifier.weight(1f),
                                    shape = CircleShape
                                ) {
                                    Icon(Icons.Default.Replay, contentDescription = null, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Tekrar Çöz")
                                }

                                Button(
                                    onClick = onBackToChapters,
                                    modifier = Modifier.weight(1f),
                                    shape = CircleShape
                                ) {
                                    Text("Bölümlere Dön")
                                }
                            }

                            OutlinedButton(
                                onClick = onBackToLibrary,
                                modifier = Modifier.fillMaxWidth(),
                                shape = CircleShape
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
                Text(
                    text = "Detaylı Soru İncelemesi",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }

            // Soru İnceleme Kartları
            itemsIndexed(result.details) { idx, detail ->
                OutlinedCard(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                    border = BorderStroke(
                        width = 1.dp,
                        color = if (detail.isCorrect) CacheHitGreen else MaterialTheme.colorScheme.error
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
                                fontWeight = FontWeight.Bold
                            )

                            if (detail.isCorrect) {
                                AssistChip(
                                    onClick = { },
                                    label = { Text("Doğru") },
                                    leadingIcon = {
                                        Icon(
                                            Icons.Default.CheckCircle,
                                            contentDescription = null,
                                            tint = CacheHitGreen,
                                            modifier = Modifier.size(16.dp)
                                        )
                                    },
                                    colors = AssistChipDefaults.assistChipColors(
                                        containerColor = CacheHitGreenContainer,
                                        labelColor = OnCacheHitGreen
                                    )
                                )
                            } else {
                                AssistChip(
                                    onClick = { },
                                    label = { Text("Yanlış") },
                                    leadingIcon = {
                                        Icon(
                                            Icons.Default.Cancel,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.error,
                                            modifier = Modifier.size(16.dp)
                                        )
                                    },
                                    colors = AssistChipDefaults.assistChipColors(
                                        containerColor = MaterialTheme.colorScheme.errorContainer,
                                        labelColor = MaterialTheme.colorScheme.onErrorContainer
                                    )
                                )
                            }
                        }

                        Text(
                            text = detail.question,
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium
                        )

                        HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))

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
                            shape = MaterialTheme.shapes.small,
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
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
