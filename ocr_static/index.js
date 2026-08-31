// ─────────────────────────────────────────────────────────────────────────────
// NHS PDF Letter Clinical OCR & Coding UI (PaddleOCR & Qwen Enabled)
// ─────────────────────────────────────────────────────────────────────────────

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const folderPathInput    = document.getElementById('folder-path-input');
const loadFolderBtn      = document.getElementById('load-folder-btn');
const recordsList        = document.getElementById('records-list');
const reviewedCountEl    = document.getElementById('reviewed-count');

const recordIndexBadge   = document.getElementById('record-index-badge');
const recordProgressText = document.getElementById('record-progress-text');
const prevRecordBtn      = document.getElementById('prev-record-btn');
const nextRecordBtn      = document.getElementById('next-record-btn');

const viewModePdfBtn     = document.getElementById('view-mode-pdf');
const viewModeTextBtn    = document.getElementById('view-mode-text');
const pdfViewerContainer = document.getElementById('pdf-viewer-container');
const textViewerContainer = document.getElementById('text-viewer-container');
const pdfViewerIframe    = document.getElementById('pdf-viewer-iframe');
const clinicalLetterText = document.getElementById('clinical-letter-text');
const highlightToggle    = document.getElementById('highlight-toggle');

const runExtractionBtn   = document.getElementById('run-extraction-btn');
const autoApproveBtn     = document.getElementById('auto-approve-btn');

const extractionEmptyState = document.getElementById('extraction-empty-state');
const entitiesWorkspace  = document.getElementById('entities-workspace');
const entitiesList       = document.getElementById('entities-list');
const categoryTabs       = document.getElementById('category-tabs');

const codingFooter       = document.getElementById('coding-footer');
const classifySnomedBtn  = document.getElementById('classify-snomed-btn');
const saveRowBtn         = document.getElementById('save-row-btn');
const exportBtn          = document.getElementById('export-btn');

const snomedModal        = document.getElementById('snomed-modal');
const snomedSearchInput  = document.getElementById('snomed-search-input');
const snomedSearchSubmit = document.getElementById('snomed-search-submit');
const snomedModalCategory = document.getElementById('snomed-modal-category');
const snomedResultsContainer = document.getElementById('snomed-results-container');

const toast = document.getElementById('toast');
const toastTitle = document.getElementById('toast-title');
const toastDesc = document.getElementById('toast-desc');
const toastIcon = document.getElementById('toast-icon');

// ── State ────────────────────────────────────────────────────────────────────
let loadedRecords = [];
let currentRecordIndex = 0;
let currentRecord = null;
let activeEntityDecisions = [];
let activeCategoryFilter = 'All';
let highlightEnabled = true;
let currentViewMode = 'pdf'; // 'pdf' or 'text'
let snomedTargetEntityId = null;

// ── Status options per category ───────────────────────────────────────────────
const STATUS_OPTIONS = {
    Diagnosis:  ['Current', 'Historical', 'Negated', 'Resolved', 'Suspected'],
    Symptom:    ['Current', 'Historical', 'Negated', 'Resolved', 'Warning', 'Side Effects'],
    Procedure:  ['Performed', 'Planned', 'Recommended', 'Monitoring'],
    Medication: ['Current', 'Started', 'Stopped', 'Changed', 'Recommended'],
    Vital:      ['N/A']
};

function getStatusOptions(category, selected) {
    const options = STATUS_OPTIONS[category] || ['Current'];
    return options.map(o =>
        `<option value="${o}"${o === selected ? ' selected' : ''}>${o}</option>`
    ).join('');
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    fetchDefaultFolder();
    attachEventListeners();
});

function attachEventListeners() {
    if (loadFolderBtn) loadFolderBtn.addEventListener('click', onLoadFolder);
    if (batchExtractBtn) batchExtractBtn.addEventListener('click', onBatchExtract);
    prevRecordBtn.addEventListener('click', () => navigateRecord(-1));
    nextRecordBtn.addEventListener('click', () => navigateRecord(1));
    runExtractionBtn.addEventListener('click', runExtraction);
    autoApproveBtn.addEventListener('click', autoApproveHighConfidence);
    classifySnomedBtn.addEventListener('click', runClassifySnomed);
    saveRowBtn.addEventListener('click', saveAndNext);
    exportBtn.addEventListener('click', exportExcel);
    highlightToggle.addEventListener('click', toggleHighlight);

    viewModePdfBtn.addEventListener('click', () => setViewMode('pdf'));
    viewModeTextBtn.addEventListener('click', () => setViewMode('text'));

    // Category tabs
    categoryTabs.querySelectorAll('.cat-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            activeCategoryFilter = tab.dataset.cat;
            categoryTabs.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            renderEntityCards();
        });
    });

    // SNOMED modal
    snomedSearchSubmit.addEventListener('click', searchSnomedTerm);
    snomedSearchInput.addEventListener('keydown', e => { if (e.key === 'Enter') searchSnomedTerm(); });
    snomedModal.addEventListener('click', e => { if (e.target === snomedModal) closeSnomedModal(); });
}

// ── Batch Extraction ──────────────────────────────────────────────────────────
async function onBatchExtract() {
    if (!confirm('Run automated 3-model extraction across ALL letters in this folder overnight?')) return;
    if (batchExtractBtn) {
        batchExtractBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Overnight Batch...';
        batchExtractBtn.disabled = true;
    }
    showToast('Overnight Batch Started', 'Processing all PDF letters automatically...', 'info');
    try {
        const res = await fetch('/api/batch-extract', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            showToast('Batch Complete ⚡', `Processed ${data.processed_count} letters overnight. Saved to Excel!`, 'success');
            await onLoadFolder();
        } else {
            showToast('Batch Error', data.detail || 'Batch processing failed.', 'danger');
        }
    } catch(e) {
        showToast('Error', 'Batch extraction request failed.', 'danger');
    } finally {
        if (batchExtractBtn) {
            batchExtractBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Batch Extract All (Overnight)';
            batchExtractBtn.disabled = false;
        }
    }
}

// ── View Mode Switcher ────────────────────────────────────────────────────────
function setViewMode(mode) {
    currentViewMode = mode;
    if (mode === 'pdf') {
        viewModePdfBtn.classList.add('active');
        viewModeTextBtn.classList.remove('active');
        pdfViewerContainer.classList.remove('hide');
        textViewerContainer.classList.add('hide');
        highlightToggle.classList.add('hide');
    } else {
        viewModeTextBtn.classList.add('active');
        viewModePdfBtn.classList.remove('active');
        textViewerContainer.classList.remove('hide');
        pdfViewerContainer.classList.add('hide');
        highlightToggle.classList.remove('hide');
    }
}

// ── Folder Loading ────────────────────────────────────────────────────────────
async function fetchDefaultFolder() {
    try {
        const res = await fetch('/api/default-folder');
        const data = await res.json();
        if (data.default_folder && folderPathInput) {
            folderPathInput.value = data.default_folder;
            // Scan folder on startup
            onLoadFolder();
        }
    } catch(e) {
        console.warn('Could not fetch default folder', e);
    }
}

async function onLoadFolder() {
    const folderPath = folderPathInput ? folderPathInput.value.trim() : '';
    if (!folderPath) {
        showToast('Warning', 'Please enter a valid folder path.', 'warning');
        return;
    }

    loadFolderBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Scanning PDFs...';
    loadFolderBtn.disabled = true;

    try {
        const res = await fetch('/api/load-folder', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder_path: folderPath })
        });
        const data = await res.json();
        if (data.success) {
            loadedRecords = data.records;
            currentRecordIndex = data.current_index || 0;
            renderRecordsSidebar();
            exportBtn.disabled = false;
            if (batchExtractBtn) batchExtractBtn.disabled = false;
            loadRecord(currentRecordIndex);
            showToast('Folder Loaded', `Found ${data.total_records} PDF letters.`, 'success');
        } else {
            showToast('Error', data.detail || 'Could not load folder.', 'danger');
        }
    } catch(e) {
        showToast('Error', 'Failed to scan folder for PDFs.', 'danger');
    } finally {
        loadFolderBtn.innerHTML = '<i class="fa-solid fa-folder-tree"></i> Load Folder PDFs';
        loadFolderBtn.disabled = false;
    }
}

// ── Sidebar Records List ──────────────────────────────────────────────────────
function renderRecordsSidebar() {
    const reviewed = loadedRecords.filter(r => r.reviewed).length;
    reviewedCountEl.textContent = `Reviewed: ${reviewed}/${loadedRecords.length}`;
    recordsList.innerHTML = '';

    loadedRecords.forEach((r, i) => {
        const div = document.createElement('div');
        div.className = `record-list-item${i === currentRecordIndex ? ' active' : ''}`;
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.justifyContent = 'space-between';
        div.style.padding = '10px 14px';
        div.style.cursor = 'pointer';
        div.style.borderBottom = '1px solid rgba(226, 232, 240, 0.6)';

        const pdfName = r.filename || `Record #${i + 1}`;
        div.innerHTML = `
            <div style="display:flex;align-items:center;gap:10px;overflow:hidden;flex:1;">
                <i class="fa-solid fa-file-pdf" style="color: #ef4444; font-size: 16px; flex-shrink: 0;"></i>
                <span style="font-size:13px;font-weight:600;color:var(--text-main);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${pdfName}">${pdfName}</span>
            </div>
            <span class="record-list-status" style="margin-left: 8px;">${r.reviewed ? '<i class="fa-solid fa-check-circle text-success" title="Reviewed"></i>' : (r.has_extractions ? '<i class="fa-solid fa-circle-dot text-warning" title="Entities Extracted"></i>' : '<i class="fa-regular fa-circle text-muted" title="Pending"></i>')}</span>
        `;
        div.addEventListener('click', () => loadRecord(i));
        recordsList.appendChild(div);
    });
}

// ── Record Loading ────────────────────────────────────────────────────────────
async function loadRecord(index) {
    currentRecordIndex = index;

    // Get item from loadedRecords array if available for fast UI update
    const item = loadedRecords[index] || {};
    const fileName = item.filename || `Record #${index + 1}`;
    recordIndexBadge.textContent = fileName;
    recordProgressText.textContent = `${index + 1} of ${loadedRecords.length}`;
    prevRecordBtn.disabled = index === 0;
    nextRecordBtn.disabled = index >= loadedRecords.length - 1;

    // Immediately load PDF into the viewer iframe (instant document view)
    pdfViewerIframe.src = `/api/view-pdf/${index}`;
    renderRecordsSidebar();

    // Show loading indicator in text container while OCR text is loaded
    clinicalLetterText.innerHTML = '<em style="color:var(--text-muted)"><i class="fa-solid fa-spinner fa-spin"></i> Extracting OCR text from PDF document...</em>';

    try {
        const res = await fetch(`/api/record/${index}`);
        currentRecord = await res.json();

        // Render extracted OCR text
        renderLetterText(currentRecord.text, []);
        highlightEnabled = true;
        highlightToggle.classList.add('active');

        // Reset entity workspace
        extractionEmptyState.classList.remove('hide');
        entitiesWorkspace.classList.add('hide');
        codingFooter.classList.add('hide');
        runExtractionBtn.disabled = false;
        autoApproveBtn.disabled = true;
        activeEntityDecisions = [];

        if (currentRecord.extracted_entities && currentRecord.extracted_entities.length > 0) {
            activeEntityDecisions = currentRecord.extracted_entities;
            showEntitiesWorkspace();
        }
    } catch(e) {
        console.error('Failed to load record details:', e);
        clinicalLetterText.innerHTML = '<em style="color:var(--danger)">Failed to fetch OCR text for this record.</em>';
    }
}

function renderLetterText(text, highlights) {
    if (!text) {
        clinicalLetterText.innerHTML = '<em style="color:var(--text-muted)">Extracting OCR text from PDF document...</em>';
        return;
    }
    if (!highlightEnabled || !highlights.length) {
        clinicalLetterText.textContent = text;
        return;
    }
    let html = escapeHtml(text);
    highlights.forEach(h => {
        if (h && h.length > 2) {
            const escaped = escapeHtml(h).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(${escaped})`, 'gi');
            html = html.replace(regex, '<mark class="highlight">$1</mark>');
        }
    });
    clinicalLetterText.innerHTML = html;
}

function toggleHighlight() {
    highlightEnabled = !highlightEnabled;
    highlightToggle.classList.toggle('active', highlightEnabled);
    refreshHighlights();
}

function refreshHighlights() {
    const terms = activeEntityDecisions
        .filter(d => d.decision === 'Yes')
        .map(d => d.text);
    renderLetterText(currentRecord?.text || '', terms);
}

// ── Extraction ────────────────────────────────────────────────────────────────
async function runExtraction() {
    runExtractionBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Qwen SLMS Extraction...';
    runExtractionBtn.disabled = true;
    try {
        const res = await fetch(`/api/extract/${currentRecordIndex}`, { method: 'POST' });
        const data = await res.json();
        if (data.entities !== undefined) {
            activeEntityDecisions = data.entities;
            if (loadedRecords[currentRecordIndex]) {
                loadedRecords[currentRecordIndex].has_extractions = true;
            }
            showEntitiesWorkspace();
            const msg = data.cached
                ? `Loaded ${data.entities.length} entities from cache instantly!`
                : `Extracted ${data.entities.length} entities with confidence status & SNOMED CT.`;
            showToast(data.cached ? 'Loaded from Cache ⚡' : 'Extraction Complete', msg, 'success');
        }
    } catch(e) {
        showToast('Error', 'Entity extraction failed.', 'danger');
    } finally {
        runExtractionBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Entity Extraction';
        runExtractionBtn.disabled = false;
    }
}

function showEntitiesWorkspace() {
    extractionEmptyState.classList.add('hide');
    entitiesWorkspace.classList.remove('hide');
    codingFooter.classList.remove('hide');
    autoApproveBtn.disabled = false;
    activeCategoryFilter = 'All';
    categoryTabs.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
    categoryTabs.querySelector('[data-cat="All"]').classList.add('active');
    renderEntityCards();
    refreshHighlights();
}

// ── Entity Cards ──────────────────────────────────────────────────────────────
function renderEntityCards() {
    entitiesList.innerHTML = '';
    const filtered = activeCategoryFilter === 'All'
        ? [...activeEntityDecisions]
        : activeEntityDecisions.filter(e => e.category === activeCategoryFilter);

    // Sort entities by confidence descending
    filtered.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));

    if (!filtered.length) {
        entitiesList.innerHTML = `<div class="list-empty-msg" style="text-align:center;padding:30px 0;color:var(--text-muted);">No ${activeCategoryFilter !== 'All' ? activeCategoryFilter : ''} entities found.</div>`;
        return;
    }

    filtered.forEach(ent => {
        const card = buildEntityCard(ent);
        entitiesList.appendChild(card);
    });
}

function buildEntityCard(ent) {
    const confVal = ent.confidence !== undefined ? parseFloat(ent.confidence) : 0.95;

    let confCls = 'confidence-low'; // RED (>70%)
    let confIcon = 'fa-circle-exclamation';
    let confLabel = '1 Model (>70%)';

    if (confVal >= 0.90) {
        confCls = 'confidence-high'; // GREEN (>90%)
        confIcon = 'fa-check-double';
        confLabel = '3 Models (>90%)';
    } else if (confVal >= 0.80) {
        confCls = 'confidence-medium'; // ORANGE (>80%)
        confIcon = 'fa-check';
        confLabel = '2 Models (>80%)';
    }

    const pct = Math.round(confVal * 100);

    const needsSnomed = ['Diagnosis', 'Symptom', 'Procedure'].includes(ent.category);
    const snomedSection = needsSnomed ? `
        <div class="snomed-link-row">
            <label><i class="fa-solid fa-code-branch"></i> SNOMED CT Code</label>
            <div class="snomed-field-wrapper">
                <input type="text" id="snomed-input-${ent.id}" class="form-control"
                    placeholder="Click search to look up SNOMED..."
                    value="${escapeHtml(ent.snomed || '')}"
                    onchange="updateEntityField('${ent.id}', 'snomed', this.value)">
                <button class="btn btn-xs btn-outline" onclick="openSnomedSearch('${ent.id}', '${ent.category}')">
                    <i class="fa-solid fa-magnifying-glass"></i> Search
                </button>
            </div>
        </div>` : '';

    const div = document.createElement('div');
    div.className = 'entity-card';
    div.dataset.id = ent.id;
    div.dataset.cat = ent.category;
    div.innerHTML = `
        <div class="entity-card-header">
            <div class="entity-title-row">
                <input type="text" class="entity-name-input" value="${escapeHtml(ent.text)}"
                    onchange="updateEntityField('${ent.id}', 'text', this.value)">
                <div class="entity-meta-row">
                    <span class="cat-badge cat-badge-${ent.category}">${ent.category}</span>
                    <span class="confidence-gauge ${confCls}" title="${confLabel}">
                        <i class="fa-solid ${confIcon}"></i> ${pct}% Confidence
                    </span>
                    <span class="status-pill" style="font-size:11px;background:rgba(15,23,42,0.05);padding:3px 8px;border-radius:4px;font-weight:600;color:#475569;">${escapeHtml(ent.validation_status || 'Validated')}</span>
                </div>
            </div>
        </div>

        <div class="decision-bar">
            <label>Coding Decision</label>
            <div class="btn-group-decision">
                <button class="btn-decision ${ent.decision === 'Yes' ? 'active-yes' : ''}"
                    onclick="setDecision('${ent.id}', 'Yes')">
                    <i class="fa-solid fa-check"></i> Include
                </button>
                <button class="btn-decision ${ent.decision === 'Do Not Need' ? 'active-skip' : ''}"
                    onclick="setDecision('${ent.id}', 'Do Not Need')">
                    <i class="fa-solid fa-minus"></i> Skip
                </button>
                <button class="btn-decision ${ent.decision === 'No' ? 'active-no' : ''}"
                    onclick="setDecision('${ent.id}', 'No')">
                    <i class="fa-solid fa-xmark"></i> Reject
                </button>
            </div>
        </div>

        <div class="entity-details-expansion${ent.decision === 'No' ? ' hide' : ''}">
            <div class="entity-detail-item">
                <label>Clinical Status</label>
                <select id="status-sel-${ent.id}" onchange="updateEntityField('${ent.id}', 'status', this.value)">
                    ${getStatusOptions(ent.category, ent.status)}
                </select>
            </div>
            <div class="entity-detail-item">
                <label>Category</label>
                <select id="cat-sel-${ent.id}" onchange="updateEntityCategory('${ent.id}', this.value)">
                    ${['Diagnosis','Symptom','Procedure','Medication','Vital'].map(c =>
                        `<option value="${c}"${c === ent.category ? ' selected':''}>${c}</option>`
                    ).join('')}
                </select>
            </div>
            ${snomedSection}
        </div>
    `;
    return div;
}

// ── Entity State Updates ──────────────────────────────────────────────────────
function setDecision(entityId, decision) {
    const ent = activeEntityDecisions.find(e => e.id === entityId);
    if (!ent) return;
    ent.decision = decision;

    const card = document.querySelector(`.entity-card[data-id="${entityId}"]`);
    if (!card) return;

    card.querySelectorAll('.btn-decision').forEach(btn => {
        btn.classList.remove('active-yes', 'active-no', 'active-skip');
    });
    const [yesBtn, skipBtn, noBtn] = card.querySelectorAll('.btn-decision');
    if (decision === 'Yes') yesBtn.classList.add('active-yes');
    else if (decision === 'Do Not Need') skipBtn.classList.add('active-skip');
    else if (decision === 'No') noBtn.classList.add('active-no');

    const expansion = card.querySelector('.entity-details-expansion');
    if (expansion) expansion.classList.toggle('hide', decision === 'No');

    refreshHighlights();
}

function updateEntityField(entityId, field, value) {
    const ent = activeEntityDecisions.find(e => e.id === entityId);
    if (ent) ent[field] = value;
}

function updateEntityCategory(entityId, newCategory) {
    const ent = activeEntityDecisions.find(e => e.id === entityId);
    if (!ent) return;
    ent.category = newCategory;
    ent.status = { Diagnosis:'Current', Symptom:'Current', Procedure:'Performed', Medication:'Current', Vital:'N/A' }[newCategory] || 'Current';

    const card = document.querySelector(`.entity-card[data-id="${entityId}"]`);
    if (card) {
        const newCard = buildEntityCard(ent);
        card.parentNode.replaceChild(newCard, card);
    }
}

// ── Auto Approve ──────────────────────────────────────────────────────────────
function autoApproveHighConfidence() {
    activeEntityDecisions.forEach(ent => {
        const target = (ent.confidence || 0.95) >= 0.80 ? 'Yes' : 'Do Not Need';
        setDecision(ent.id, target);
    });
    showToast('Auto-coded', 'High confidence entities included, others skipped.', 'success');
}

// ── Classify & SNOMED ─────────────────────────────────────────────────────────
async function runClassifySnomed() {
    classifySnomedBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Mapping SNOMED...';
    classifySnomedBtn.disabled = true;
    try {
        const payload = {
            row_index: currentRecordIndex,
            decisions: activeEntityDecisions
        };
        const res = await fetch('/api/process-row', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            data.processed_entities.forEach(pe => {
                if (pe.decision !== 'Yes') return;
                const ent = activeEntityDecisions.find(e => e.id === pe.id);
                if (ent) { ent.snomed = pe.snomed; ent.status = pe.status; }
                const snomedInput = document.getElementById(`snomed-input-${pe.id}`);
                if (snomedInput) snomedInput.value = pe.snomed || '';
            });
            if (loadedRecords[currentRecordIndex]) {
                loadedRecords[currentRecordIndex].reviewed = true;
            }
            renderRecordsSidebar();
            showToast('Mapped', 'SNOMED codes mapped and saved for this PDF.', 'success');
        }
    } catch(e) {
        showToast('Error', 'Classification/SNOMED mapping failed.', 'danger');
    } finally {
        classifySnomedBtn.innerHTML = '<i class="fa-solid fa-gears"></i> Classify &amp; Map SNOMED';
        classifySnomedBtn.disabled = false;
    }
}

// ── Save & Next ───────────────────────────────────────────────────────────────
async function saveAndNext() {
    saveRowBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';
    saveRowBtn.disabled = true;
    try {
        const payload = {
            row_index: currentRecordIndex,
            decisions: activeEntityDecisions
        };
        const res = await fetch('/api/process-row', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            if (loadedRecords[currentRecordIndex]) loadedRecords[currentRecordIndex].reviewed = true;
            renderRecordsSidebar();
            showToast('Saved', `PDF record #${currentRecordIndex + 1} saved.`, 'success');
            if (currentRecordIndex < loadedRecords.length - 1) {
                setTimeout(() => loadRecord(currentRecordIndex + 1), 400);
            }
        }
    } catch(e) {
        showToast('Error', 'Failed to save record.', 'danger');
    } finally {
        saveRowBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save &amp; Next PDF';
        saveRowBtn.disabled = false;
    }
}

function navigateRecord(offset) {
    const next = currentRecordIndex + offset;
    if (next >= 0 && next < loadedRecords.length) loadRecord(next);
}

// ── Export ────────────────────────────────────────────────────────────────────
async function exportExcel() {
    exportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Exporting...';
    exportBtn.disabled = true;
    try {
        const res = await fetch('/api/export', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            const link = document.createElement('a');
            link.href = data.download_url;
            link.setAttribute('download', data.filename);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast('Exported', 'Spreadsheet downloaded successfully.', 'success');
        }
    } catch(e) {
        showToast('Error', 'Export failed.', 'danger');
    } finally {
        exportBtn.innerHTML = '<i class="fa-solid fa-file-export"></i> Export Final Spreadsheet';
        exportBtn.disabled = false;
    }
}

// ── SNOMED Modal ──────────────────────────────────────────────────────────────
window.openSnomedSearch = function(entityId, category) {
    snomedTargetEntityId = entityId;
    snomedModalCategory.textContent = category;
    snomedSearchInput.value = '';
    snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg">Type a search term and click search.</div>';
    snomedModal.classList.remove('hide');
    snomedSearchInput.focus();
};

window.closeSnomedModal = function() {
    snomedModal.classList.add('hide');
    snomedTargetEntityId = null;
};

async function searchSnomedTerm() {
    const term = snomedSearchInput.value.trim();
    if (term.length < 2) {
        snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg" style="color:var(--danger)">Enter at least 2 characters.</div>';
        return;
    }
    snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg"><i class="fa-solid fa-spinner fa-spin"></i> Searching SNOMED CT...</div>';
    try {
        const category = snomedModalCategory.textContent;
        const res = await fetch(`/api/snomed-search?term=${encodeURIComponent(term)}&category=${encodeURIComponent(category)}`);
        const data = await res.json();
        if (data.success && data.results && data.results.length > 0) {
            snomedResultsContainer.innerHTML = '';
            data.results.forEach(item => {
                const div = document.createElement('div');
                div.className = 'snomed-result-item';
                div.innerHTML = `
                    <div class="snomed-result-info">
                        <span class="snomed-result-display">${escapeHtml(item.snomed_display)}</span>
                        <span class="snomed-result-code">Code: ${item.snomed_code}</span>
                    </div>
                    <button class="btn btn-xs btn-primary">Select</button>
                `;
                div.addEventListener('click', () => selectSnomedCode(item.snomed_code, item.snomed_display));
                snomedResultsContainer.appendChild(div);
            });
        } else {
            snomedResultsContainer.innerHTML = `<div class="snomed-empty-msg">No concepts found for "${escapeHtml(term)}".</div>`;
        }
    } catch(e) {
        snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg" style="color:var(--danger)">Error querying SNOMED CT server.</div>';
    }
}

function selectSnomedCode(code, display) {
    if (snomedTargetEntityId) {
        const ent = activeEntityDecisions.find(e => e.id === snomedTargetEntityId);
        const formatted = `${code} | ${display}`;
        if (ent) ent.snomed = formatted;
        const inp = document.getElementById(`snomed-input-${snomedTargetEntityId}`);
        if (inp) inp.value = formatted;
    }
    closeSnomedModal();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(title, desc, type = 'success') {
    toastTitle.textContent = title;
    toastDesc.textContent = desc;
    if (type === 'danger') {
        toastIcon.className = 'fa-solid fa-circle-exclamation';
        toastIcon.style.color = 'var(--danger)';
        toast.style.borderColor = 'rgba(239,68,68,0.4)';
    } else {
        toastIcon.className = 'fa-solid fa-circle-check';
        toastIcon.style.color = 'var(--success)';
        toast.style.borderColor = 'rgba(16,185,129,0.4)';
    }
    toast.classList.remove('hide');
    setTimeout(() => toast.classList.add('hide'), 4000);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
