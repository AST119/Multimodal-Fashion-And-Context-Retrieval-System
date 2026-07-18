
const queryInput = document.getElementById('query-input');
const searchBtn = document.getElementById('search-btn');
const indexBtn = document.getElementById('index-btn');
const resultsGrid = document.getElementById('results-grid');
const statusMessage = document.getElementById('status-message');

async function runSearch() {
    const query = queryInput.value.trim();
    if (!query) return;

    resultsGrid.innerHTML = '<div class="placeholder-text">Searching database...</div>';
    showStatus("Executing visual-semantic search...", false);

    try {
        const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        renderResults(data.results);
        hideStatus();
    } catch (error) {
        console.error(error);
        showStatus("Error executing search.", true);
    }
}

function renderResults(results) {
    if (!results || results.length === 0) {
        resultsGrid.innerHTML = '<div class="placeholder-text">No matching records found.</div>';
        return;
    }

    resultsGrid.innerHTML = '';
    results.forEach(item => {
        const card = document.createElement('div');
        card.className = 'fashion-card';

        let tagsHtml = '';
        item.matched_garments.forEach(g => {
            tagsHtml += `<span class="tag">${g.precise_color} (${g.broad_color}) ${g.category}</span>`;
        });

        if (tagsHtml === '') {
            tagsHtml = '<span class="tag">Scene Match</span>';
        }

        card.innerHTML = `
            <div class="image-wrapper">
                <img src="/image?path=${encodeURIComponent(item.image_path)}" alt="Fashion">
                <div class="score-badge">Match: ${(item.score * 100).toFixed(1)}%</div>
            </div>
            <div class="card-details">
                <h3>${item.image_name}</h3>
                <div class="tags-list">${tagsHtml}</div>
            </div>
        `;
        resultsGrid.appendChild(card);
    });
}

async function triggerIndexing() {
    showStatus("Triggering localized indexing process (CPU). Please wait...", false);
    indexBtn.disabled = true;

    try {
        const response = await fetch('/index', { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') {
            showStatus("Indexing complete! Database successfully updated.", false);
        } else {
            showStatus(`Indexing failed: ${data.detail}`, true);
        }
    } catch (error) {
        showStatus("Error executing local indexing.", true);
    } finally {
        indexBtn.disabled = false;
    }
}

function showStatus(text, isError = false) {
    statusMessage.innerText = text;
    statusMessage.className = "status-box";
    if (isError) {
        statusMessage.style.backgroundColor = "#fee2e2";
        statusMessage.style.borderColor = "#fca5a5";
        statusMessage.style.color = "#991b1b";
    } else {
        statusMessage.style.backgroundColor = "#eff6ff";
        statusMessage.style.borderColor = "#bfdbfe";
        statusMessage.style.color = "#1e3a8a";
    }
}

function hideStatus() {
    statusMessage.className = "status-box hidden";
}

searchBtn.addEventListener('click', runSearch);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') runSearch();
});
indexBtn.addEventListener('click', triggerIndexing);