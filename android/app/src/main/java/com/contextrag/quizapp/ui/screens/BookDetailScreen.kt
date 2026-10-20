package com.contextrag.quizapp.ui.screens

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoStories
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.data.ChapterItem
import com.contextrag.quizapp.ui.components.SectionHeader
import com.contextrag.quizapp.ui.components.ShimmerCard
import com.contextrag.quizapp.ui.components.expressiveBounceClick
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.M3ExpressiveMotion
import com.contextrag.quizapp.ui.theme.M3Shapes
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
import com.contextrag.quizapp.ui.theme.titleLargeEmphasized
import com.contextrag.quizapp.ui.theme.titleMediumEmphasized
import com.contextrag.quizapp.viewmodel.UiState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BookDetailScreen(
    state: UiState,
    onBack: () -> Unit,
    onRefresh: () -> Unit,
    onSelectChapter: (ChapterItem) -> Unit,
    onSetDifficulty: (String) -> Unit,
    onSetCount: (Int) -> Unit,
    onStartQuiz: () -> Unit
) {
    val book = state.selectedBook
    if (book == null) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Button(
                onClick = onBack,
                shape = M3Shapes.CornerFull,
                modifier = Modifier.expressiveBounceClick(onClick = onBack)
            ) {
                Text("Kitaplığa Dön")
            }
        }
        return
    }

    val hasChapters = state.chaptersData?.chapters?.isNotEmpty() == true

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = book.title,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        style = MaterialTheme.typography.titleMediumEmphasized
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Geri")
                    }
                },
                actions = {
                    IconButton(onClick = onRefresh) {
                        Icon(Icons.Default.Refresh, contentDescription = "Yenile")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        },
        bottomBar = {
            // Yalnızca bölümler mevcut olduğunda sabit M3 Expressive alt paneli göster
            if (hasChapters) {
                Surface(
                    tonalElevation = 6.dp,
                    shadowElevation = 8.dp,
                    shape = M3Shapes.CornerExtraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 14.dp, vertical = 10.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = state.selectedChapter?.title ?: "Lütfen Bir Bölüm Seçin",
                                style = MaterialTheme.typography.titleMediumEmphasized,
                                color = MaterialTheme.colorScheme.primary,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f)
                            )
                            if (state.selectedChapter != null) {
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
                                            text = "Havuz: ${state.selectedChapter.poolCount}",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = OnCacheHitGreen,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }
                        }

                        // Zorluk & Soru Sayısı Seçenekleri
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            // Zorluk
                            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                listOf("kolay", "orta", "zor").forEach { diff ->
                                    FilterChip(
                                        selected = state.quizDifficulty == diff,
                                        onClick = { onSetDifficulty(diff) },
                                        label = { Text(diff.replaceFirstChar { it.uppercase() }) },
                                        shape = M3Shapes.CornerFull,
                                        modifier = Modifier.expressiveBounceClick(scaleDown = 0.94f) {
                                            onSetDifficulty(diff)
                                        }
                                    )
                                }
                            }

                            // Soru Sayısı
                            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                listOf(3, 5, 10).forEach { cnt ->
                                    FilterChip(
                                        selected = state.quizCount == cnt,
                                        onClick = { onSetCount(cnt) },
                                        label = { Text("$cnt") },
                                        shape = M3Shapes.CornerFull,
                                        modifier = Modifier.expressiveBounceClick(scaleDown = 0.94f) {
                                            onSetCount(cnt)
                                        }
                                    )
                                }
                            }
                        }

                        // M3 Expressive Quiz Başlat Butonu
                        Button(
                            onClick = onStartQuiz,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp)
                                .expressiveBounceClick(scaleDown = 0.96f) {
                                    if (state.selectedChapter != null && !state.isLoadingQuiz) {
                                        onStartQuiz()
                                    }
                                },
                            enabled = state.selectedChapter != null && !state.isLoadingQuiz,
                            shape = M3Shapes.CornerFull,
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.primary,
                                contentColor = MaterialTheme.colorScheme.onPrimary
                            )
                        ) {
                            if (state.isLoadingQuiz) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = MaterialTheme.colorScheme.onPrimary,
                                    strokeWidth = 2.dp
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Soru Bankasından Çekiliyor...")
                            } else {
                                Icon(Icons.Default.PlayArrow, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Bölüm Quizini Başlat", fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }
            }
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            contentPadding = PaddingValues(top = 10.dp, bottom = if (hasChapters) 24.dp else 48.dp)
        ) {
            // 1. Kitap Başlık & Bilgi Kartı (M3 Hero Card with Nested Shapes)
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = M3Shapes.CornerLargeIncreased,
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(18.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = book.title,
                            style = MaterialTheme.typography.titleLargeEmphasized,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Text(
                            text = "${book.author ?: "Bilinmiyor"} • ID: #${book.bookId}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        if (!book.subjects.isNullOrBlank()) {
                            Text(
                                text = book.subjects,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.outline,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                        }

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(top = 4.dp)
                        ) {
                            if (book.isCachedLocally) {
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
                                            imageVector = Icons.Default.Bolt,
                                            contentDescription = null,
                                            tint = OnCacheHitGreen,
                                            modifier = Modifier.size(12.dp)
                                        )
                                        Text(
                                            text = "Cihazda Hazır",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = OnCacheHitGreen,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }

                            if (state.isLoadingChapters) {
                                Surface(
                                    shape = M3Shapes.CornerFull,
                                    color = MaterialTheme.colorScheme.secondaryContainer
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(10.dp),
                                            strokeWidth = 2.dp
                                        )
                                        Text("Taranıyor...", style = MaterialTheme.typography.labelSmall)
                                    }
                                }
                            } else if (state.chaptersError != null) {
                                Surface(
                                    shape = M3Shapes.CornerFull,
                                    color = MaterialTheme.colorScheme.errorContainer
                                ) {
                                    Text(
                                        text = "⚠️ Metin Erişilemedi",
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onErrorContainer,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            } else if (state.chaptersData != null) {
                                Surface(
                                    shape = M3Shapes.CornerFull,
                                    color = MaterialTheme.colorScheme.secondaryContainer
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.AutoStories,
                                            contentDescription = null,
                                            modifier = Modifier.size(12.dp),
                                            tint = MaterialTheme.colorScheme.onSecondaryContainer
                                        )
                                        Text(
                                            text = "${state.chaptersData.totalChapters} Bölüm",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSecondaryContainer,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 2. Yükleniyor Durumu (Shimmer Cards)
            if (state.isLoadingChapters) {
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(
                            text = "Bölümler taranıyor ve metin doğrulanıyor...",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        ShimmerCard(height = 76.dp)
                        ShimmerCard(height = 76.dp)
                        ShimmerCard(height = 76.dp)
                    }
                }
            }

            // 3. Hata Durumu (In-Screen Error Card with Bounce Buttons)
            if (state.chaptersError != null && !state.isLoadingChapters) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = M3Shapes.CornerLarge,
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.5f)
                        ),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.4f))
                    ) {
                        Column(
                            modifier = Modifier.padding(20.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Icon(
                                Icons.Default.ErrorOutline,
                                contentDescription = null,
                                modifier = Modifier.size(44.dp),
                                tint = MaterialTheme.colorScheme.error
                            )
                            Text(
                                text = "Kitap Metnine Ulaşılamadı",
                                style = MaterialTheme.typography.titleMediumEmphasized,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Text(
                                text = state.chaptersError,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                textAlign = TextAlign.Center
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                OutlinedButton(
                                    onClick = onRefresh,
                                    modifier = Modifier
                                        .weight(1f)
                                        .expressiveBounceClick(onClick = onRefresh),
                                    shape = M3Shapes.CornerFull
                                ) {
                                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Tekrar Dene")
                                }
                                Button(
                                    onClick = onBack,
                                    modifier = Modifier
                                        .weight(1f)
                                        .expressiveBounceClick(onClick = onBack),
                                    shape = M3Shapes.CornerFull
                                ) {
                                    Icon(Icons.Default.Search, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Kitap Ara")
                                }
                            }
                        }
                    }
                }
            }

            // 4. Bölümler Listesi (Animated Selection & Spring Progress Bar)
            if (state.chaptersData != null && !state.isLoadingChapters) {
                item {
                    SectionHeader(
                        title = "Bölüm Seçimi & Okuma Durumu",
                        badgeText = "${state.chaptersData.chapters.size} Bölüm"
                    )
                }

                items(state.chaptersData.chapters) { chapter ->
                    val isSelected = state.selectedChapter?.chapterIndex == chapter.chapterIndex
                    val animatedBorderWidth by animateDpAsState(
                        targetValue = if (isSelected) 2.dp else 1.dp,
                        animationSpec = M3ExpressiveMotion.fastSpatial(),
                        label = "border_width"
                    )
                    val animatedBorderColor by animateColorAsState(
                        targetValue = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant,
                        animationSpec = M3ExpressiveMotion.defaultEffects(),
                        label = "border_color"
                    )
                    val animatedContainerColor by animateColorAsState(
                        targetValue = if (isSelected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f)
                        else MaterialTheme.colorScheme.surfaceContainerLow,
                        animationSpec = M3ExpressiveMotion.defaultEffects(),
                        label = "container_color"
                    )
                    val animatedProgress by animateFloatAsState(
                        targetValue = (chapter.progressPercent / 100f).coerceIn(0f, 1f),
                        animationSpec = M3ExpressiveMotion.defaultSpatial(),
                        label = "progress"
                    )

                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .expressiveBounceClick(scaleDown = 0.97f) { onSelectChapter(chapter) },
                        shape = M3Shapes.CornerMedium,
                        border = BorderStroke(animatedBorderWidth, animatedBorderColor),
                        colors = CardDefaults.cardColors(containerColor = animatedContainerColor)
                    ) {
                        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    modifier = Modifier.weight(1f)
                                ) {
                                    RadioButton(
                                        selected = isSelected,
                                        onClick = { onSelectChapter(chapter) }
                                    )
                                    Text(
                                        text = chapter.title,
                                        style = if (isSelected) MaterialTheme.typography.titleMediumEmphasized
                                        else MaterialTheme.typography.titleSmall,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }

                                Surface(
                                    shape = M3Shapes.CornerFull,
                                    color = if (chapter.poolCount > 0) CacheHitGreenContainer else MaterialTheme.colorScheme.surfaceContainerHigh
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.Bolt,
                                            contentDescription = null,
                                            modifier = Modifier.size(13.dp),
                                            tint = if (chapter.poolCount > 0) OnCacheHitGreen else MaterialTheme.colorScheme.outline
                                        )
                                        Text(
                                            text = "Havuz: ${chapter.poolCount}",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = if (chapter.poolCount > 0) OnCacheHitGreen else MaterialTheme.colorScheme.outline,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }

                            // Spring Animated LinearProgressIndicator
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(start = 12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                LinearProgressIndicator(
                                    progress = animatedProgress,
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(6.dp),
                                    trackColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = "%${chapter.progressPercent.toInt()}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.outline
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
