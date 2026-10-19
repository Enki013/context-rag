package com.contextrag.quizapp.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoStories
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.data.ChapterItem
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
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
            Button(onClick = onBack, shape = CircleShape) {
                Text("Kitaplığa Dön")
            }
        }
        return
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = book.title,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
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
            // Sabit Alt Panel: Quiz Ayarları & Başlat Butonu
            Surface(
                tonalElevation = 6.dp,
                shadowElevation = 8.dp,
                shape = MaterialTheme.shapes.extraLarge,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp)
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
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )
                        if (state.selectedChapter != null) {
                            SuggestionChip(
                                onClick = {},
                                label = { Text("Havuz: ${state.selectedChapter.poolCount}") },
                                icon = { Icon(Icons.Default.Bolt, contentDescription = null, modifier = Modifier.size(14.dp)) },
                                colors = SuggestionChipDefaults.suggestionChipColors(
                                    containerColor = CacheHitGreenContainer,
                                    labelColor = OnCacheHitGreen
                                ),
                                border = null
                            )
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
                                    label = { Text(diff.replaceFirstChar { it.uppercase() }) }
                                )
                            }
                        }

                        // Soru Sayısı
                        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            listOf(3, 5, 10).forEach { cnt ->
                                FilterChip(
                                    selected = state.quizCount == cnt,
                                    onClick = { onSetCount(cnt) },
                                    label = { Text("$cnt") }
                                )
                            }
                        }
                    }

                    // Quiz Başlat Butonu
                    Button(
                        onClick = onStartQuiz,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = state.selectedChapter != null && !state.isLoadingQuiz,
                        shape = CircleShape
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
                            Text("Bölüm Quizini Başlat")
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
            verticalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(top = 8.dp, bottom = 16.dp)
        ) {
            // Kitap Başlık & Bilgi Kartı
            item {
                ElevatedCard(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                    colors = CardDefaults.elevatedCardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            text = book.title,
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
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
                            modifier = Modifier.padding(top = 4.dp)
                        ) {
                            if (book.isCachedLocally) {
                                SuggestionChip(
                                    onClick = {},
                                    label = { Text("⚡ Cihazda Hazır") },
                                    colors = SuggestionChipDefaults.suggestionChipColors(
                                        containerColor = CacheHitGreenContainer,
                                        labelColor = OnCacheHitGreen
                                    ),
                                    border = null
                                )
                            }
                            SuggestionChip(
                                onClick = {},
                                label = { Text("${state.chaptersData?.totalChapters ?: 0} Bölüm") },
                                icon = { Icon(Icons.Default.AutoStories, contentDescription = null, modifier = Modifier.size(14.dp)) }
                            )
                        }
                    }
                }
            }

            item {
                Text(
                    text = "Bölüm Seçimi & Okuma Durumu",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }

            // Yükleniyor durumu
            if (state.isLoadingChapters) {
                item {
                    Box(modifier = Modifier.fillMaxWidth().padding(32.dp), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            CircularProgressIndicator()
                            Text("Bölümler taranıyor...", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            } else if (state.chaptersData != null) {
                items(state.chaptersData.chapters) { chapter ->
                    val isSelected = state.selectedChapter?.chapterIndex == chapter.chapterIndex
                    OutlinedCard(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onSelectChapter(chapter) },
                        shape = MaterialTheme.shapes.medium,
                        border = BorderStroke(
                            width = if (isSelected) 2.dp else 1.dp,
                            color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                        ),
                        colors = CardDefaults.outlinedCardColors(
                            containerColor = if (isSelected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f)
                            else MaterialTheme.colorScheme.surface
                        )
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
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }

                                AssistChip(
                                    onClick = { onSelectChapter(chapter) },
                                    label = { Text("Havuz: ${chapter.poolCount}") },
                                    leadingIcon = { Icon(Icons.Default.Bolt, contentDescription = null, modifier = Modifier.size(14.dp)) }
                                )
                            }

                            // İlerleme Çubuğu
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(start = 12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                LinearProgressIndicator(
                                    progress = (chapter.progressPercent / 100f).coerceIn(0f, 1f),
                                    modifier = Modifier.weight(1f)
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
