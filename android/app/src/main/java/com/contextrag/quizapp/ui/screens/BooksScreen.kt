package com.contextrag.quizapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.AutoStories
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.data.BookItem
import com.contextrag.quizapp.ui.components.MetricCard
import com.contextrag.quizapp.ui.components.SectionHeader
import com.contextrag.quizapp.ui.components.ShimmerCard
import com.contextrag.quizapp.ui.components.expressiveBounceClick
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.M3Shapes
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
import com.contextrag.quizapp.ui.theme.titleMediumEmphasized
import com.contextrag.quizapp.viewmodel.UiState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BooksScreen(
    state: UiState,
    onSearchChange: (String) -> Unit,
    onSearchSubmit: (String) -> Unit,
    onSelectBook: (BookItem) -> Unit
) {
    val totalIndexed = state.health?.totalBooks ?: 79446

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(top = 16.dp, bottom = 96.dp)
    ) {
        // 1. M3 Metric Cards Özeti (Personal Records Tarzı)
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MetricCard(
                    title = "Katalog",
                    value = "${totalIndexed / 1000}k+",
                    badgeText = "FTS5",
                    note = "İndeksli Kitap",
                    valueIcon = Icons.AutoMirrored.Filled.MenuBook,
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Hızlı Servis",
                    value = "0ms",
                    badgeText = "Cache",
                    note = "Hazır Soru Havuzu",
                    valueIcon = Icons.Default.Bolt,
                    valueIconTint = CacheHitGreen,
                    badgeBg = CacheHitGreenContainer,
                    badgeColor = OnCacheHitGreen,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        // 2. M3 Expressive Arama Kutusu & Hızlı Filtreler
        item {
            OutlinedTextField(
                value = state.searchQuery,
                onValueChange = onSearchChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Kitap, yazar veya konu ara...") },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = "Ara") },
                trailingIcon = {
                    if (state.searchQuery.isNotEmpty()) {
                        IconButton(onClick = { onSearchChange("") }) {
                            Icon(Icons.Default.Clear, contentDescription = "Temizle")
                        }
                    }
                },
                singleLine = true,
                shape = M3Shapes.CornerExtraLarge,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainerLow.copy(alpha = 0.6f),
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outlineVariant
                )
            )

            Spacer(modifier = Modifier.height(10.dp))

            // Hızlı Arama Önerileri (M3 Filter Chips with Expressive Bounce)
            val quickQueries = listOf("Sherlock Holmes", "Jane Austen", "Frankenstein", "Nutuk", "Felsefe")
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(quickQueries) { q ->
                    FilterChip(
                        selected = state.searchQuery == q,
                        onClick = {
                            onSearchChange(q)
                            onSearchSubmit(q)
                        },
                        label = { Text(q) },
                        shape = M3Shapes.CornerFull,
                        modifier = Modifier.expressiveBounceClick(scaleDown = 0.94f) {
                            onSearchChange(q)
                            onSearchSubmit(q)
                        }
                    )
                }
            }
        }

        // 3. Arama Yükleniyor Durumu (Smooth Shimmer Placeholders)
        if (state.isSearching) {
            item {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    ShimmerCard(height = 84.dp)
                    ShimmerCard(height = 84.dp)
                    ShimmerCard(height = 84.dp)
                }
            }
        }

        // 4. Kitap Arama Sonuçları
        if (state.searchResults.isNotEmpty()) {
            item {
                SectionHeader(
                    title = "Bulunan Kitaplar",
                    badgeText = "${state.searchResults.size} Kitap"
                )
            }

            items(state.searchResults) { book ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .expressiveBounceClick(scaleDown = 0.97f) { onSelectBook(book) },
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                    ),
                    shape = M3Shapes.CornerLarge
                ) {
                    ListItem(
                        headlineContent = {
                            Text(
                                text = book.title,
                                style = MaterialTheme.typography.titleMediumEmphasized,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                        },
                        supportingContent = {
                            Column(
                                modifier = Modifier.padding(top = 4.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "${book.author ?: "Bilinmeyen Yazar"} • ID: #${book.bookId}",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                if (!book.subjects.isNullOrBlank()) {
                                    Text(
                                        text = book.subjects,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.outline,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                                if (book.isCachedLocally) {
                                    Surface(
                                        shape = M3Shapes.CornerFull,
                                        color = CacheHitGreenContainer,
                                        modifier = Modifier.padding(top = 2.dp)
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
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
                                                text = "Cihazda Hazır (0ms)",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = OnCacheHitGreen,
                                                fontWeight = FontWeight.Bold
                                            )
                                        }
                                    }
                                }
                            }
                        },
                        leadingContent = {
                            Surface(
                                shape = M3Shapes.CornerMedium,
                                color = if (book.isCachedLocally) CacheHitGreenContainer
                                else MaterialTheme.colorScheme.secondaryContainer,
                                modifier = Modifier.size(44.dp)
                            ) {
                                Box(contentAlignment = Alignment.Center) {
                                    Icon(
                                        imageVector = if (book.isCachedLocally) Icons.Default.AutoStories else Icons.AutoMirrored.Filled.MenuBook,
                                        contentDescription = null,
                                        tint = if (book.isCachedLocally) OnCacheHitGreen else MaterialTheme.colorScheme.onSecondaryContainer,
                                        modifier = Modifier.size(22.dp)
                                    )
                                }
                            }
                        },
                        trailingContent = {
                            IconButton(onClick = { onSelectBook(book) }) {
                                Icon(
                                    Icons.Default.ChevronRight,
                                    contentDescription = "Bölümleri Gör",
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        },
                        colors = ListItemDefaults.colors(
                            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                        )
                    )
                }
            }
        } else if (!state.isSearching) {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 48.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Surface(
                            shape = M3Shapes.CornerLarge,
                            color = MaterialTheme.colorScheme.surfaceContainerHigh,
                            modifier = Modifier.size(72.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(
                                    Icons.AutoMirrored.Filled.MenuBook,
                                    contentDescription = null,
                                    modifier = Modifier.size(36.dp),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                        Text(
                            text = "Aramak için bir kitap veya konu seçin",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}
