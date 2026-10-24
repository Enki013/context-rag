package com.contextrag.quizapp.ui.screens

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Quiz
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.ui.components.expressiveBounceClick
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.M3ExpressiveMotion
import com.contextrag.quizapp.ui.theme.M3Shapes
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
import com.contextrag.quizapp.ui.theme.titleMediumEmphasized
import com.contextrag.quizapp.viewmodel.UiState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun QuizScreen(
    state: UiState,
    onSelectOption: (questionId: Int, option: String) -> Unit,
    onNextQuestion: () -> Unit,
    onPrevQuestion: () -> Unit,
    onSubmitQuiz: () -> Unit,
    onExitQuiz: () -> Unit
) {
    val quiz = state.currentQuiz

    if (quiz == null || quiz.questions.isEmpty()) {
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
                                Icons.Default.Quiz,
                                contentDescription = null,
                                modifier = Modifier.size(36.dp),
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                    Text(
                        text = "Aktif Quiz Bulunmuyor",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Lütfen önce bir kitap ve bölüm seçerek 'Bölüm Quizini Başlat' butonuna dokunun.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Button(
                        onClick = onExitQuiz,
                        shape = M3Shapes.CornerFull,
                        modifier = Modifier.expressiveBounceClick(onClick = onExitQuiz)
                    ) {
                        Text("Bölüm Seçimine Dön")
                    }
                }
            }
        }
        return
    }

    var showExitDialog by remember { mutableStateOf(false) }
    var showSubmitDialog by remember { mutableStateOf(false) }

    // Geri tuşuna basıldığında onay diyalogu göster
    BackHandler {
        showExitDialog = true
    }

    val currentIdx = state.currentQuestionIndex.coerceIn(0, quiz.questions.size - 1)
    val totalQuestions = quiz.questions.size
    val answeredCount = state.selectedAnswers.size

    val animatedProgress by animateFloatAsState(
        targetValue = (currentIdx + 1).toFloat() / totalQuestions,
        animationSpec = M3ExpressiveMotion.defaultSpatial(),
        label = "quiz_progress"
    )

    // Sınavdan Çıkış Onay Diyaloğu
    if (showExitDialog) {
        AlertDialog(
            onDismissRequest = { showExitDialog = false },
            icon = { Icon(Icons.Default.Close, contentDescription = null, tint = MaterialTheme.colorScheme.error) },
            title = { Text("Sınavdan Çıkılsın mı?") },
            text = {
                Text("Sınavı iptal etmek üzeresiniz. Kaydedilmemiş cevaplarınız silinecektir ve bölüm detayına döneceksiniz.")
            },
            shape = M3Shapes.CornerExtraLargeIncreased,
            confirmButton = {
                Button(
                    onClick = {
                        showExitDialog = false
                        onExitQuiz()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                    shape = M3Shapes.CornerFull,
                    modifier = Modifier.expressiveBounceClick {
                        showExitDialog = false
                        onExitQuiz()
                    }
                ) {
                    Text("Sınavı Bitir ve Çık")
                }
            },
            dismissButton = {
                OutlinedButton(
                    onClick = { showExitDialog = false },
                    shape = M3Shapes.CornerFull,
                    modifier = Modifier.expressiveBounceClick { showExitDialog = false }
                ) {
                    Text("Devam Et")
                }
            }
        )
    }

    // Tamamlama & Değerlendirme Onay Diyaloğu
    if (showSubmitDialog) {
        AlertDialog(
            onDismissRequest = { showSubmitDialog = false },
            icon = { Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary) },
            title = { Text("Quizi Tamamla") },
            text = {
                Text("Toplam $totalQuestions sorudan $answeredCount tanesini cevapladınız. Cevaplarınızı sunucuda değerlendirmek istiyor musunuz?")
            },
            shape = M3Shapes.CornerExtraLargeIncreased,
            confirmButton = {
                Button(
                    onClick = {
                        showSubmitDialog = false
                        onSubmitQuiz()
                    },
                    shape = M3Shapes.CornerFull,
                    modifier = Modifier.expressiveBounceClick {
                        showSubmitDialog = false
                        onSubmitQuiz()
                    }
                ) {
                    Text("Değerlendir")
                }
            },
            dismissButton = {
                OutlinedButton(
                    onClick = { showSubmitDialog = false },
                    shape = M3Shapes.CornerFull,
                    modifier = Modifier.expressiveBounceClick { showSubmitDialog = false }
                ) {
                    Text("Sorulara Dön")
                }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = state.selectedChapter?.title ?: "Bölüm Quizi",
                            style = MaterialTheme.typography.titleMediumEmphasized,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = state.selectedBook?.title ?: "",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = { showExitDialog = true }) {
                        Icon(Icons.Default.Close, contentDescription = "Sınavdan Çık")
                    }
                },
                actions = {
                    Surface(
                        shape = M3Shapes.CornerFull,
                        color = MaterialTheme.colorScheme.secondaryContainer,
                        modifier = Modifier.padding(end = 16.dp)
                    ) {
                        Text(
                            text = "${currentIdx + 1} / $totalQuestions",
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // İlerleme Çubuğu & Rozetler (Spring Animated)
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                LinearProgressIndicator(
                    progress = { animatedProgress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp),
                    trackColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                    color = MaterialTheme.colorScheme.primary
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (quiz.cacheHit) {
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
                                    Icons.Default.Bolt,
                                    contentDescription = null,
                                    tint = OnCacheHitGreen,
                                    modifier = Modifier.size(13.dp)
                                )
                                Text(
                                    text = "⚡ Soru Havuzu (Cache Hit)",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = OnCacheHitGreen,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    } else {
                        Surface(
                            shape = M3Shapes.CornerFull,
                            color = MaterialTheme.colorScheme.surfaceContainerHigh
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    Icons.Default.Psychology,
                                    contentDescription = null,
                                    modifier = Modifier.size(13.dp),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = "🤖 Qwen 2.5 LLM",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }

                    Text(
                        text = "Zorluk: ${state.quizDifficulty.replaceFirstChar { it.uppercase() }}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            }

            // Soru ve Şıklar (AnimatedContent with M3 Spatial Springs)
            AnimatedContent(
                targetState = currentIdx,
                transitionSpec = {
                    val isForward = targetState > initialState
                    val enterSlide = slideInHorizontally(
                        animationSpec = M3ExpressiveMotion.defaultSpatial(),
                        initialOffsetX = { fullWidth -> if (isForward) fullWidth / 2 else -fullWidth / 2 }
                    ) + fadeIn(animationSpec = M3ExpressiveMotion.fastEffects())

                    val exitSlide = slideOutHorizontally(
                        animationSpec = M3ExpressiveMotion.defaultSpatial(),
                        targetOffsetX = { fullWidth -> if (isForward) -fullWidth / 2 else fullWidth / 2 }
                    ) + fadeOut(animationSpec = M3ExpressiveMotion.fastEffects())

                    enterSlide togetherWith exitSlide
                },
                label = "question_transition"
            ) { qIdx ->
                val question = quiz.questions[qIdx]
                val selectedOption = state.selectedAnswers[question.id]

                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    // Soru Kartı
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = M3Shapes.CornerLargeIncreased,
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
                    ) {
                        Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(
                                text = "Soru ${qIdx + 1}",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = question.question,
                                style = MaterialTheme.typography.titleMediumEmphasized
                            )
                            if (!question.sourceChapter.isNullOrBlank()) {
                                Text(
                                    text = "Kaynak: ${question.sourceChapter}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.outline
                                )
                            }
                        }
                    }

                    // Şıklar (M3 Cards with Spring Bounce & Animated Selection)
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        question.options.forEach { option ->
                            val isSelected = selectedOption == option
                            val animatedBorderWidth by animateDpAsState(
                                targetValue = if (isSelected) 2.dp else 1.dp,
                                animationSpec = M3ExpressiveMotion.fastSpatial(),
                                label = "option_border_width"
                            )
                            val animatedBorderColor by animateColorAsState(
                                targetValue = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant,
                                animationSpec = M3ExpressiveMotion.defaultEffects(),
                                label = "option_border_color"
                            )
                            val animatedContainerColor by animateColorAsState(
                                targetValue = if (isSelected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f)
                                else MaterialTheme.colorScheme.surfaceContainerLow,
                                animationSpec = M3ExpressiveMotion.defaultEffects(),
                                label = "option_container_color"
                            )

                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .expressiveBounceClick(scaleDown = 0.98f) {
                                        onSelectOption(question.id, option)
                                    },
                                shape = M3Shapes.CornerMedium,
                                border = BorderStroke(animatedBorderWidth, animatedBorderColor),
                                colors = CardDefaults.cardColors(containerColor = animatedContainerColor)
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 16.dp, vertical = 14.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = isSelected,
                                        onClick = { onSelectOption(question.id, option) }
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
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Alt Navigasyon Butonları
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 32.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = onPrevQuestion,
                    enabled = currentIdx > 0,
                    shape = M3Shapes.CornerFull,
                    modifier = Modifier.expressiveBounceClick(onClick = onPrevQuestion)
                ) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Önceki")
                }

                if (currentIdx < totalQuestions - 1) {
                    FilledTonalButton(
                        onClick = onNextQuestion,
                        shape = M3Shapes.CornerFull,
                        modifier = Modifier.expressiveBounceClick(onClick = onNextQuestion)
                    ) {
                        Text("Sonraki")
                        Spacer(modifier = Modifier.width(4.dp))
                        Icon(Icons.AutoMirrored.Filled.ArrowForward, contentDescription = null, modifier = Modifier.size(18.dp))
                    }
                } else {
                    Button(
                        onClick = { showSubmitDialog = true },
                        shape = M3Shapes.CornerFull,
                        enabled = !state.isEvaluating,
                        modifier = Modifier.expressiveBounceClick { showSubmitDialog = true }
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
}
