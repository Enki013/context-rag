package com.contextrag.quizapp.ui

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MenuBook
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.ui.components.expressiveBounceClick
import com.contextrag.quizapp.ui.screens.BookDetailScreen
import com.contextrag.quizapp.ui.screens.BooksScreen
import com.contextrag.quizapp.ui.screens.QuizScreen
import com.contextrag.quizapp.ui.screens.ResultScreen
import com.contextrag.quizapp.ui.screens.SettingsScreen
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.M3ExpressiveMotion
import com.contextrag.quizapp.ui.theme.M3Shapes
import com.contextrag.quizapp.ui.theme.titleLargeEmphasized
import com.contextrag.quizapp.viewmodel.AppScreen
import com.contextrag.quizapp.viewmodel.QuizViewModel
import kotlinx.coroutines.flow.collectLatest

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: QuizViewModel) {
    val state by viewModel.uiState.collectAsState()
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(Unit) {
        viewModel.snackbarEvent.collectLatest { message ->
            snackbarHostState.showSnackbar(message)
        }
    }

    // Ana kütüphanede değilsek donanım/hareket geri tuşu hiyerarşik geri gitsin
    BackHandler(enabled = state.currentScreen != AppScreen.LIBRARY) {
        viewModel.navigateBack()
    }

    // Kök sekmeler: Yalnızca LIBRARY veya SETTINGS durumunda ana Scaffold çubukları gösterilir
    val isRootScreen = state.currentScreen == AppScreen.LIBRARY || state.currentScreen == AppScreen.SETTINGS

    Scaffold(
        topBar = {
            if (isRootScreen) {
                CenterAlignedTopAppBar(
                    title = {
                        Text(
                            text = if (state.currentScreen == AppScreen.LIBRARY) "Context RAG Kitaplık" else "Sunucu & Ayarlar",
                            style = MaterialTheme.typography.titleLargeEmphasized
                        )
                    },
                    actions = {
                        val isOnline = state.health?.status == "ok"
                        Surface(
                            shape = M3Shapes.CornerFull,
                            color = MaterialTheme.colorScheme.surfaceContainerHigh,
                            modifier = Modifier.padding(end = 16.dp)
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(8.dp)
                                        .clip(CircleShape)
                                        .background(if (isOnline) CacheHitGreen else MaterialTheme.colorScheme.error)
                                )
                                Text(
                                    text = if (isOnline) "Bağlı" else "Çevrimdışı",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }
                    },
                    colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    )
                )
            }
        },
        bottomBar = {
            if (isRootScreen) {
                NavigationBar(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                    tonalElevation = 3.dp
                ) {
                    NavigationBarItem(
                        selected = state.currentScreen == AppScreen.LIBRARY,
                        onClick = { viewModel.navigateTo(AppScreen.LIBRARY) },
                        icon = { Icon(Icons.Default.MenuBook, contentDescription = "Kitaplık") },
                        label = { Text("Kitaplık") },
                        modifier = Modifier.expressiveBounceClick(scaleDown = 0.94f) {
                            viewModel.navigateTo(AppScreen.LIBRARY)
                        }
                    )
                    NavigationBarItem(
                        selected = state.currentScreen == AppScreen.SETTINGS,
                        onClick = { viewModel.navigateTo(AppScreen.SETTINGS) },
                        icon = { Icon(Icons.Default.Settings, contentDescription = "Sunucu") },
                        label = { Text("Sunucu") },
                        modifier = Modifier.expressiveBounceClick(scaleDown = 0.94f) {
                            viewModel.navigateTo(AppScreen.SETTINGS)
                        }
                    )
                }
            }
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(if (isRootScreen) innerPadding else PaddingValues(0.dp))
        ) {
            AnimatedContent(
                targetState = state.currentScreen,
                transitionSpec = {
                    val isForward = targetState.ordinal > initialState.ordinal
                    val enter = slideInHorizontally(
                        animationSpec = M3ExpressiveMotion.defaultSpatial(),
                        initialOffsetX = { fullWidth -> if (isForward) fullWidth / 3 else -fullWidth / 3 }
                    ) + fadeIn(animationSpec = M3ExpressiveMotion.fastEffects())

                    val exit = slideOutHorizontally(
                        animationSpec = M3ExpressiveMotion.defaultSpatial(),
                        targetOffsetX = { fullWidth -> if (isForward) -fullWidth / 3 else fullWidth / 3 }
                    ) + fadeOut(animationSpec = M3ExpressiveMotion.fastEffects())

                    enter togetherWith exit
                },
                label = "ScreenTransition"
            ) { screen ->
                when (screen) {
                    AppScreen.LIBRARY -> BooksScreen(
                        state = state,
                        onSearchChange = viewModel::onSearchQueryChanged,
                        onSearchSubmit = viewModel::searchBooks,
                        onSelectBook = viewModel::selectBook
                    )
                    AppScreen.BOOK_DETAIL -> BookDetailScreen(
                        state = state,
                        onBack = viewModel::navigateBack,
                        onRefresh = viewModel::refreshChapters,
                        onSelectChapter = viewModel::selectChapter,
                        onSetDifficulty = viewModel::setDifficulty,
                        onSetCount = viewModel::setQuizCount,
                        onStartQuiz = viewModel::startQuiz
                    )
                    AppScreen.QUIZ -> QuizScreen(
                        state = state,
                        onSelectOption = viewModel::onOptionSelected,
                        onNextQuestion = viewModel::nextQuestion,
                        onPrevQuestion = viewModel::prevQuestion,
                        onSubmitQuiz = viewModel::submitQuiz,
                        onExitQuiz = viewModel::returnToChapters
                    )
                    AppScreen.RESULT -> ResultScreen(
                        state = state,
                        onRetryQuiz = viewModel::startQuiz,
                        onBackToChapters = viewModel::returnToChapters,
                        onBackToLibrary = viewModel::returnToLibrary
                    )
                    AppScreen.SETTINGS -> SettingsScreen(
                        state = state,
                        onUpdateBaseUrl = viewModel::updateBaseUrl,
                        onRefreshHealth = viewModel::checkHealth
                    )
                }
            }
        }
    }
}
