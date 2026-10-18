package com.contextrag.quizapp.ui

import androidx.compose.animation.Crossfade
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.EmojiEvents
import androidx.compose.material.icons.filled.MenuBook
import androidx.compose.material.icons.filled.Quiz
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.ui.screens.BooksScreen
import com.contextrag.quizapp.ui.screens.QuizScreen
import com.contextrag.quizapp.ui.screens.ResultScreen
import com.contextrag.quizapp.ui.screens.SettingsScreen
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.viewmodel.QuizViewModel
import kotlinx.coroutines.flow.collectLatest

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: QuizViewModel) {
    val state by viewModel.uiState.collectAsState()
    val snackbarHostState = remember { SnackbarHostState() }
    var currentTab by remember { mutableIntStateOf(0) }

    LaunchedEffect(Unit) {
        viewModel.snackbarEvent.collectLatest { message ->
            snackbarHostState.showSnackbar(message)
        }
    }

    // Quiz başlatıldığında otomatik Quiz sekmesine geç
    LaunchedEffect(state.currentQuiz) {
        if (state.currentQuiz != null && state.evaluationResult == null) {
            currentTab = 1
        }
    }

    // Değerlendirme sonucu geldiğinde otomatik Sonuçlar sekmesine geç
    LaunchedEffect(state.evaluationResult) {
        if (state.evaluationResult != null) {
            currentTab = 2
        }
    }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "Context RAG Quiz",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                actions = {
                    // Sunucu Durum Noktası (Yeşil / Kırmızı)
                    val isOnline = state.health?.status == "ok"
                    Row(
                        modifier = Modifier.padding(end = 16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(CircleShape)
                                .background(if (isOnline) CacheHitGreen else MaterialTheme.colorScheme.error)
                        )
                        Text(
                            text = if (isOnline) "Bağlı" else "Çevrimdışı",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        },
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surfaceContainer,
                tonalElevation = 3.dp
            ) {
                NavigationBarItem(
                    selected = currentTab == 0,
                    onClick = { currentTab = 0 },
                    icon = { Icon(Icons.Default.MenuBook, contentDescription = "Kitaplar") },
                    label = { Text("Kitaplar") }
                )
                NavigationBarItem(
                    selected = currentTab == 1,
                    onClick = { currentTab = 1 },
                    icon = {
                        BadgedBox(
                            badge = {
                                if (state.currentQuiz != null && state.evaluationResult == null) {
                                    Badge { Text("${state.currentQuiz?.questionCount}") }
                                }
                            }
                        ) {
                            Icon(Icons.Default.Quiz, contentDescription = "Quiz")
                        }
                    },
                    label = { Text("Quiz") }
                )
                NavigationBarItem(
                    selected = currentTab == 2,
                    onClick = { currentTab = 2 },
                    icon = { Icon(Icons.Default.EmojiEvents, contentDescription = "Sonuçlar") },
                    label = { Text("Sonuçlar") }
                )
                NavigationBarItem(
                    selected = currentTab == 3,
                    onClick = { currentTab = 3 },
                    icon = { Icon(Icons.Default.Settings, contentDescription = "Sunucu") },
                    label = { Text("Sunucu") }
                )
            }
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            Crossfade(targetState = currentTab, label = "TabCrossfade") { tab ->
                when (tab) {
                    0 -> BooksScreen(
                        state = state,
                        onSearchChange = viewModel::onSearchQueryChanged,
                        onSearchSubmit = viewModel::searchBooks,
                        onSelectBook = viewModel::selectBook,
                        onSelectChapter = viewModel::selectChapter,
                        onSetDifficulty = viewModel::setDifficulty,
                        onSetCount = viewModel::setQuizCount,
                        onStartQuiz = viewModel::startQuiz
                    )
                    1 -> QuizScreen(
                        state = state,
                        onSelectOption = viewModel::onOptionSelected,
                        onNextQuestion = viewModel::nextQuestion,
                        onPrevQuestion = viewModel::prevQuestion,
                        onSubmitQuiz = viewModel::submitQuiz,
                        onNavigateToBooks = { currentTab = 0 }
                    )
                    2 -> ResultScreen(
                        state = state,
                        onRetryQuiz = {
                            viewModel.startQuiz()
                            currentTab = 1
                        },
                        onBackToBooks = {
                            viewModel.resetQuiz()
                            currentTab = 0
                        }
                    )
                    3 -> SettingsScreen(
                        state = state,
                        onUpdateBaseUrl = viewModel::updateBaseUrl,
                        onRefreshHealth = viewModel::checkHealth
                    )
                }
            }
        }
    }
}
