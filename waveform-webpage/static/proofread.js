import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'

document.addEventListener('DOMContentLoaded', function () {
    // --- 元素获取 (已恢复所有原始元素) ---
    const statusDisplay = document.getElementById('statusDisplay');
    const correctorUI = document.getElementById('corrector-ui');
    const videoElement = document.getElementById('videoPlayer');
    const currentSentenceTitle = document.getElementById('current-sentence-title');
    const startTimeInput = document.getElementById('start-time');
    const endTimeInput = document.getElementById('end-time');
    const playSelectionBtn = document.getElementById('play-selection-btn');
    const saveBtn = document.getElementById('save-btn');
    const sentenceSelectorCard = document.getElementById('sentence-selector-card');
    const sentenceList = document.getElementById('sentence-list');
    // 【功能恢复】获取手动上传相关元素
    const manualUploadCard = document.getElementById('manual-upload-card');
    const loadManualBtn = document.getElementById('load-manual-files-btn');
    const manualVideoInput = document.getElementById('manualVideoFile');
    const manualJsonInput = document.getElementById('manualJsonFile');

    let wavesurfer = null;
    let regionsPlugin = null;
    let activeRegion = null;
    let analysisData = null; 
    let currentClipId = null;

    // --- 主初始化函数 ---
    async function init() {
        const params = new URLSearchParams(window.location.search);
        const videoUrl = params.get('video_url');
        const jsonUrl = params.get('json_url');

        if (videoUrl && jsonUrl) {
            await loadDataAndSetupEditor(videoUrl, jsonUrl);
        } else {
            // 【功能恢复】如果URL参数不存在，则显示手动上传卡片
            updateStatus('错误：缺少文件链接。请手动上传文件。', 'error');
            manualUploadCard.classList.remove('hidden');
        }
    }

    // --- 数据加载与编辑器设置 ---
    async function loadDataAndSetupEditor(videoUrl, jsonUrl) {
        try {
            updateStatus('正在加载分析结果...', 'loading');
            
            let jsonData;
            // 如果是文件对象，直接读取；如果是URL，则fetch
            if (typeof jsonUrl === 'string') {
                const response = await fetch(jsonUrl);
                if (!response.ok) throw new Error(`无法加载JSON文件 (${response.status})`);
                jsonData = await response.json();
            } else {
                jsonData = jsonUrl; // 已经是解析好的JSON对象
            }
            analysisData = jsonData; // 保存到全局

            if (!analysisData.clips || analysisData.clips.length === 0) {
                throw new Error('分析结果中不包含任何有效的句子片段。');
            }

            updateStatus('数据加载成功，正在初始化播放器...', 'loading');
            await setupWaveSurfer(videoUrl); 
            
            populateSentenceSelector(analysisData.clips);
            sentenceSelectorCard.classList.remove('hidden');
            manualUploadCard.classList.add('hidden'); // 成功后隐藏手动上传
            updateStatus('初始化完成。请从下方列表选择一个句子开始校对。', 'success');

        } catch (error) {
            console.error('加载或设置时出错:', error);
            updateStatus(`加载数据失败: ${error.message}`, 'error');
            manualUploadCard.classList.remove('hidden'); // 出错时也显示手动上传
        }
    }
    
    // --- 初始化 WaveSurfer ---
    function setupWaveSurfer(mediaUrl) {
        return new Promise((resolve, reject) => {
            if (wavesurfer) wavesurfer.destroy();
            
            videoElement.src = mediaUrl;

            regionsPlugin = RegionsPlugin.create();
            wavesurfer = WaveSurfer.create({
                container: '#waveform',
                waveColor: 'rgb(135, 168, 206)',
                progressColor: 'rgb(43, 83, 131)',
                media: videoElement,
                // 【功能恢复】恢复原始代码中的滚动和缩放选项
                autoScroll: true,
                minPxPerSec: 100,
                plugins: [
                    regionsPlugin,
                    TimelinePlugin.create({ container: '#timeline' })
                ],
            });

            regionsPlugin.on('region-updated', (region) => {
                activeRegion = region;
                updateTimestampInputs();
            });
            
            let isPlayingRegion = false;
            regionsPlugin.on('region-out', () => { if (isPlayingRegion) wavesurfer.pause(); });
            wavesurfer.on('pause', () => { isPlayingRegion = false; });
            playSelectionBtn.onclick = () => {
                if (activeRegion) {
                    isPlayingRegion = true;
                    activeRegion.play();
                }
            };

            // 【功能恢复】手动修改输入框时，更新 region
            startTimeInput.addEventListener('change', () => {
                if (activeRegion) activeRegion.setOptions({ start: parseFloat(startTimeInput.value) });
            });
            endTimeInput.addEventListener('change', () => {
                if (activeRegion) activeRegion.setOptions({ end: parseFloat(endTimeInput.value) });
            });

            wavesurfer.on('ready', resolve);
            wavesurfer.on('error', reject);
        });
    }

    // --- 填充句子选择列表 ---
    function populateSentenceSelector(clips) {
        sentenceList.innerHTML = '';
        clips.forEach(clip => {
            const item = document.createElement('div');
            item.className = 'sentence-item';
            item.id = `item-${clip.id}`;
            item.innerHTML = `
                <span class="sentence-text">[${clip.score.toFixed(0)}分] ${clip.sentence}</span>
                <span class="sentence-status" id="status-${clip.id}"></span>
            `;
            item.addEventListener('click', () => {
                document.querySelectorAll('.sentence-item.active').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                setupCorrectorForClip(clip);
            });
            sentenceList.appendChild(item);
        });
    }
    
    // --- 为选定片段设置校对器 ---
    function setupCorrectorForClip(clip) {
        currentClipId = clip.id;
        correctorUI.classList.remove('hidden');
        currentSentenceTitle.textContent = `正在校对: "${clip.sentence}"`;
        regionsPlugin.clearRegions();
        activeRegion = regionsPlugin.addRegion({
            id: clip.id,
            start: clip.predicted_start,
            end: clip.predicted_end,
            color: 'rgba(246, 173, 85, 0.25)',
            drag: true,
            resize: true,
        });
        updateTimestampInputs();
        wavesurfer.seekTo(activeRegion.start / wavesurfer.getDuration());
    }

    // --- 更新时间输入框 ---
    function updateTimestampInputs() {
        if (activeRegion) {
            startTimeInput.value = activeRegion.start.toFixed(3);
            endTimeInput.value = activeRegion.end.toFixed(3);
        }
    }

    // --- 保存按钮逻辑 ---
    saveBtn.addEventListener('click', async () => {
        if (!currentClipId || !activeRegion) {
            alert('请先从列表中选择一个句子进行校对。');
            return;
        }
        const statusElement = document.getElementById(`status-${currentClipId}`);
        statusElement.textContent = '裁剪中...';
        saveBtn.disabled = true;
        const payload = {
            start_time: parseFloat(startTimeInput.value),
            end_time: parseFloat(endTimeInput.value),
            clip_source_path: analysisData.clip_source_path,
            original_video_filename: analysisData.original_video_filename,
            sentence_id: currentClipId
        };
        try {
            const response = await fetch('/clip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || '裁剪失败');
            statusElement.innerHTML = `<a href="${result.download_url}" class="download-link" target="_blank" download>下载</a>`;
        } catch (error) {
            console.error('裁剪请求失败:', error);
            statusElement.textContent = '裁剪失败';
            statusElement.style.color = 'red';
        } finally {
            saveBtn.disabled = false;
        }
    });
    
    // --- 【功能恢复】手动加载逻辑 ---
    loadManualBtn.addEventListener('click', () => {
        const videoFile = manualVideoInput.files[0];
        const jsonFile = manualJsonInput.files[0];

        if (!videoFile || !jsonFile) {
            alert('请同时选择视频文件和JSON文件。');
            return;
        }
        
        const videoUrl = URL.createObjectURL(videoFile);
        const jsonReader = new FileReader();
        
        jsonReader.onload = (event) => {
            try {
                const jsonData = JSON.parse(event.target.result);
                // 直接调用主设置函数，传递文件对象和解析后的JSON
                loadDataAndSetupEditor(videoUrl, jsonData);
            } catch (e) {
                updateStatus(`手动加载JSON失败: ${e.message}`, 'error');
            }
        };
        jsonReader.readAsText(jsonFile);
    });

    // --- 工具函数: 更新状态显示 ---
    function updateStatus(message, type) {
        statusDisplay.textContent = message;
        statusDisplay.classList.remove('loading', 'success', 'error');
        if (type === 'loading') statusDisplay.style.backgroundColor = '#e2e8f0';
        else if (type === 'success') statusDisplay.style.backgroundColor = '#c6f6d5';
        else if (type === 'error') statusDisplay.style.backgroundColor = '#fed7d7';
    }

    // 启动程序
    init();
});
