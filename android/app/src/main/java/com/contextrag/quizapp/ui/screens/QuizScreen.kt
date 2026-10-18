package com.contextrag.quizapp.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Quiz
import androidx.compose.material3.*
import androidx.compose.runtime.*
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
fun QuizScreen(
    state: UiState,
    onSelectOption: (questionId: Int, option: String) -> Unit,
    onNextQuestion: () -> Unit,
    onPrevQuestion: () -> Unit,
    onSubmitQuiz: () -> Unit,
    onNavigateToBooks: () -> Unit
) {
    val quiz = state.currentQuiz

    if (quiz == null || quiz.questions.isEmpty()) {
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
                        Icons.Default.Quiz,
                        contentDescription = null,
                        modifier = Modifier.size(56.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "Aktif Quiz Bulunmuyor",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Lütfen önce 'Kitaplar' sekmesinden bir kitap ve bölüm seçerek 'Quiz Başlat' butonuna dokunun.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Button(
                        onClick = onNavigateToBooks,
                        shape = MaterialTheme.shapes.full
                    ) {
                        Text("Kitap ve Bölüm Seç")
                    }
                }
            }
        }
        return
    }

    var showConfirmDialog by remember { mutableStateOf(false) }
    val currentIdx = state.currentQuestionIndex.coerceIn(0, quiz.questions.size - 1)
    val currentQuestion = quiz.questions[currentIdx]
    val selectedOption = state.selectedAnswers[currentQuestion.id]
    val totalQuestions = quiz.questions.size
    val answeredCount = state.selectedAnswers.size

    if (showConfirmDialog) {
        AlertDialog(
            onDismissRequest = { showConfirmDialog = false },
            title = { Text("Quizi Tamamla") },
            text = {
                Text("Toplam $totalQuestions sorudan $answeredCount tanesini cevapladınız. Cevaplarınızı sunucuda değerlendirmek istiyor musunuz?")
            },
            confirmButton = {
                Button(onClick = {
                    showConfirmDialog = false
                    onSubmitQuiz()
                }) {
                    Text("Değerlendir")
                }
            },
            dismissButton = {
                OutlinedButton(onClick = { showConfirmDialog = false }) {
                    Text("Devam Et")
                }
            }
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Üst Bilgi Rozetleri & Cache Hit Göstergesi
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Cache Hit Rozeti
            if (quiz.cacheHit) {
                SuggestionChip(
                    onClick = { },
                    label = { Text("⚡ 0ms Soru Bankası (Cache Hit)") },
                    icon = { Icon(Icons.Default.Bolt, contentDescription = null, modifier = Modifier.size(16.dp)) },
                    colors = SuggestionChipDefaults.suggestionChipColors(
                        containerColor = CacheHitGreenContainer,
                        labelColor = OnCacheHitGreen,
                        iconContentColor = CacheHitGreen
                    ),
                    border = BorderStroke(1.dp, CacheHitGreen)
                )
            } else {
                SuggestionChip(
                    onClick = { },
                    label = { Text("🤖 LLM Üretimi") },
                    icon = { Icon(Icons.Default.Psychology, contentDescription = null, modifier = Modifier.size(16.dp)) }
                )
            }

            // Soru Sayacı
            Badge(
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                contentColor = MaterialTheme.colorScheme.onSecondaryContainer
            ) {
                Text(
                    text = "${currentIdx + 1} / $totalQuestions",
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // İlerleme Çubuğu
        LinearProgressIndicator(
            progress = (currentIdx + 1).toFloat() / totalQuestions,
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp),
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
            color = MaterialTheme.colorScheme.primary
        )

        // Soru Kartı (M3 ElevatedCard)
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large,
            elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = "Soru ${currentIdx + 1}",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = currentQuestion.question,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium
                )
                if (!currentQuestion.sourceChapter.isNullOrBlank()) {
                    Text(
                        text = "Kaynak: ${currentQuestion.sourceChapter}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            }
        }

        // Şıklar (M3 OutlinedCards with RadioButtons)
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            currentQuestion.options.forEach { option ->
                val isSelected = selectedOption == option
                OutlinedCard(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelectOption(currentQuestion.id, option) },
                    shape = MaterialTheme.shapes.medium,
                    border = BorderStroke(
                        width = if (isSelected) 2.dp else 1.dp,
                        color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                    ),
                    colors = CardDefaults.outlinedCardColors(
                        containerColor = if (isSelected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)
                        else MaterialTheme.colorScheme.surface
                    )
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = isSelected,
                            onClick = { onSelectOption(currentQuestion.id, option) }
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = option,
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
                            color = if (isSelected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Alt Navigasyon Butonları
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 80.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(
                onClick = onPrevQuestion,
                enabled = currentIdx > 0,
                shape = MaterialTheme.shapes.full
            ) {
                Icon(Icons.Default.ArrowBack, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(4.dp))
                Text("Önceki")
            }

            if (currentIdx < totalQuestions - 1) {
                FilledTonalButton(
                    onClick = onNextQuestion,
                    shape = MaterialTheme.shapes.full
                ) {
                    Text("Sonraki")
                    Spacer(modifier = Modifier.width(4.dp))
                    Icon(Icons.Default.ArrowForward, contentDescription = null, modifier = Modifier.size(18.dp))
                }
            } else {
                Button(
                    onClick = { showConfirmDialog = true },
                    shape = MaterialTheme.shapes.full,
                    enabled = !state.isEvaluating
                ) {
                    if (state.isEvaluating) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Değerlendiriliyor...")
                    } else {
                        Icon(Icons.Default.CheckCircle, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Testi Bitir")
                    }
                }
            }
        }
    }
}
