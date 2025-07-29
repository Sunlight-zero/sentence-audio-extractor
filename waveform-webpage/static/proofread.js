// static/proofread.js

// 导入库的方式与原始文件完全一致
import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'

document.addEventListener('DOMContentLoaded', function () {
    // --- 元素获取 ---
    const statusDisplay = document.getElementById('statusDisplay');
    const correctorUI = document.getElementById('corrector-ui');
    const videoElement = document.getElementById('videoPlayer');
    const sentenceDisplay = document.getElementById('target-sentence-display');
    const startTimeInput = document.getElementById('start-time');
    const endTimeInput = document.getElementById('end-time');
    const playSelectionBtn = document.getElementById('play-selection-btn');
    const saveBtn = document.getElementById('save-btn');
    const manualUploadCard = document.getElementById('manual-upload-card');
    const loadManualBtn = document.getElementById('load-manual-files-btn');
    const manualVideoInput = document.getElementById('manualVideoFile');
    const manualJsonInput = document.getElementById('manualJsonFile');

    let wavesurfer = null;
    let activeRegion = null;
    let analysisData = null; // 用于存储从JSON加载的数据

    // --- 主初始化函数 ---
    function init() {
        const params = new URLSearchParams(window.location.search);
        const videoUrl = params.get('video_url');
        const jsonUrl = params.get('json_url');

        if (videoUrl && jsonUrl) {
            loadDataAndSetupEditor(videoUrl, jsonUrl);
        } else {
            statusDisplay.textContent = '错误：缺少必要的文件链接。';
            statusDisplay.style.backgroundColor = '#fed7d7'; // 红色背景
            manualUploadCard.classList.remove('hidden');
        }
    }

    // --- 数据加载与编辑器设置 ---
    async function loadDataAndSetupEditor(videoUrl, jsonUrl) {
        try {
            const response = await fetch(jsonUrl);
            if (!response.ok) throw new Error(`无法加载JSON文件 (${response.status})`);
            analysisData = await response.json();

            statusDisplay.textContent = '数据加载成功，正在初始化校对工具...';
            statusDisplay.style.backgroundColor = '#c6f6d5'; // 绿色背景
            
            // 使用加载的数据设置UI
            setupCorrectorUI(analysisData, videoUrl);

        } catch (error) {
            console.error('加载数据失败:', error);
            statusDisplay.textContent = `加载数据失败: ${error.message}`;
            statusDisplay.style.backgroundColor = '#fed7d7';
            manualUploadCard.classList.remove('hidden');
        }
    }

    /**
     * 设置校对UI，此函数逻辑严格遵循原始 script.js 的写法
     * @param {object} data - 从JSON文件加载的数据
     * @param {string} videoUrl - 视频文件的URL
     */
    function setupCorrectorUI(data, videoUrl) {
        correctorUI.classList.remove('hidden');
        sentenceDisplay.textContent = data.sentence;
        videoElement.src = videoUrl;

        if (wavesurfer) {
            wavesurfer.destroy();
        }

        // 插件创建方式与原始代码一致
        const regions = RegionsPlugin.create();
        const timeline = TimelinePlugin.create({
            container: '#timeline',
            formatTimeCallback: (seconds) => {
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = (seconds % 60).toFixed(0);
                return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
            },
        });

        // WaveSurfer 初始化方式与原始代码一致
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgb(135, 168, 206)',
            progressColor: 'rgb(43, 83, 131)',
            media: videoElement,
            autoScroll: true,
            minPxPerSec: 100,
            plugins: [regions, timeline],
        });

        // 关键：使用 'decode' 事件监听，与原始代码保持一致
        wavesurfer.on('decode', () => {
            regions.clearRegions();
            activeRegion = regions.addRegion({
                start: data.predicted_start,
                end: data.predicted_end,
                color: 'rgba(246, 173, 85, 0.25)',
                drag: true,
                resize: true,
            });
            updateTimestampInputs();
        });

        // 其他事件监听也与原始代码保持一致
        regions.on('region-updated', (region) => {
            activeRegion = region;
            updateTimestampInputs();
        });

        let isPlayingRegion = false;

        regions.on('region-out', () => {
            if (isPlayingRegion) {
                wavesurfer.pause();
            }
        });
        
        wavesurfer.on('pause', () => {
            isPlayingRegion = false;
        });

        wavesurfer.on('finish', () => {
            isPlayingRegion = false;
        });
        
        playSelectionBtn.onclick = () => {
            if (activeRegion) {
                isPlayingRegion = true;
                // 原始代码中没有 await，这里也保持一致
                wavesurfer.play(activeRegion.start, activeRegion.end);
            }
        };
    }

    function updateTimestampInputs() {
        if (activeRegion) {
            startTimeInput.value = activeRegion.start.toFixed(3);
            endTimeInput.value = activeRegion.end.toFixed(3);
        }
    }

    // --- 保存按钮逻辑 ---
    saveBtn.addEventListener('click', async () => {
        if (!activeRegion || !analysisData) {
            alert('没有可保存的选区或数据。');
            return;
        }

        statusDisplay.textContent = '正在保存校对结果并裁剪...';
        statusDisplay.style.backgroundColor = '#e2e8f0';
        saveBtn.disabled = true;

        const payload = {
            start_time: parseFloat(startTimeInput.value),
            end_time: parseFloat(endTimeInput.value),
            clip_source_path: analysisData.clip_source_path,
            original_video_filename: analysisData.original_video_filename
        };

        try {
            const response = await fetch('/clip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || '裁剪失败');

            statusDisplay.textContent = '裁剪成功！点击下方链接下载。';
            statusDisplay.style.backgroundColor = '#c6f6d5';

            // 创建并显示下载链接
            const downloadLink = document.createElement('a');
            downloadLink.href = result.download_url;
            downloadLink.textContent = `下载裁剪后的音频 (${result.download_url.split('/').pop()})`;
            downloadLink.className = 'button'; // 复用样式
            downloadLink.style.display = 'block';
            downloadLink.style.marginTop = '1rem';
            downloadLink.download = true;
            correctorUI.appendChild(downloadLink);

        } catch (error) {
            console.error('裁剪请求失败:', error);
            statusDisplay.textContent = `保存错误: ${error.message}`;
            statusDisplay.style.backgroundColor = '#fed7d7';
        } finally {
            saveBtn.disabled = false;
        }
    });

    // --- 手动加载逻辑 ---
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
                analysisData = jsonData; // 保存数据
                statusDisplay.textContent = '手动加载文件成功，正在初始化...';
                statusDisplay.style.backgroundColor = '#c6f6d5';
                manualUploadCard.classList.add('hidden');
                setupCorrectorUI(jsonData, videoUrl);
            } catch (e) {
                statusDisplay.textContent = `手动加载JSON失败: ${e.message}`;
                statusDisplay.style.backgroundColor = '#fed7d7';
            }
        };
        jsonReader.readAsText(jsonFile);
    });

    // 启动程序
    init();
});
