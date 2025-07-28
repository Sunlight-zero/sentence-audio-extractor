// static/script.js

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'

document.addEventListener('DOMContentLoaded', function () {
    const API_BASE_URL = 'http://127.0.0.1:5000';

    // 获取 HTML 元素
    const videoFileInput = document.getElementById('videoFile');
    const sentenceInput = document.getElementById('targetSentence');
    const startTaskBtn = document.getElementById('startTaskBtn');
    const statusDisplay = document.getElementById('statusDisplay');
    const correctorUI = document.getElementById('corrector-ui');
    const videoElement = document.getElementById('videoPlayer');
    const startTimeInput = document.getElementById('start-time');
    const endTimeInput = document.getElementById('end-time');
    const playSelectionBtn = document.getElementById('play-selection-btn');
    const saveBtn = document.getElementById('save-btn');

    let wavesurfer = null;
    let activeRegion = null;
    let pollingInterval = null;

    startTaskBtn.addEventListener('click', async () => {
        const videoFile = videoFileInput.files[0];
        const sentence = sentenceInput.value.trim();

        if (!videoFile || !sentence) {
            alert('请选择一个视频文件并填写目标句子！');
            return;
        }

        correctorUI.classList.add('hidden');
        if (pollingInterval) clearInterval(pollingInterval);
        statusDisplay.textContent = '正在上传文件并启动任务...';
        statusDisplay.style.backgroundColor = '#e2e8f0';
        startTaskBtn.disabled = true;

        const formData = new FormData();
        formData.append('videoFile', videoFile);
        formData.append('sentence', sentence);

        try {
            const response = await fetch(`${API_BASE_URL}/upload_and_start_task`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '启动任务失败');
            
            statusDisplay.textContent = '后端已接收任务，处理中... (这可能需要几分钟)';
            startPolling();
        } catch (error) {
            statusDisplay.textContent = `错误: ${error.message}`;
            statusDisplay.style.backgroundColor = '#fed7d7';
            startTaskBtn.disabled = false;
        }
    });

    function startPolling() {
        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/get_task_result`);
                const data = await response.json();

                if (data.status === 'pending_correction') {
                    clearInterval(pollingInterval);
                    statusDisplay.textContent = 'AI 处理完成，请开始校对！';
                    statusDisplay.style.backgroundColor = '#c6f6d5';
                    startTaskBtn.disabled = false;
                    setupCorrectorUI(data);
                } else if (data.status === 'error') {
                    clearInterval(pollingInterval);
                    statusDisplay.textContent = `处理失败: ${data.error}`;
                    statusDisplay.style.backgroundColor = '#fed7d7';
                    startTaskBtn.disabled = false;
                } else {
                    statusDisplay.textContent += '.';
                }
            } catch (error) {
                clearInterval(pollingInterval);
                statusDisplay.textContent = `轮询错误: ${error.message}`;
                statusDisplay.style.backgroundColor = '#fed7d7';
                startTaskBtn.disabled = false;
            }
        }, 3000);
    }

    function setupCorrectorUI(data) {
        correctorUI.classList.remove('hidden');
        videoElement.src = API_BASE_URL + data.video_url;

        if (wavesurfer) {
            wavesurfer.destroy();
        }

        const regions = RegionsPlugin.create();
        const timeline = TimelinePlugin.create({
            container: '#timeline',
            formatTimeCallback: (seconds) => {
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = (seconds % 60).toFixed(0);
                return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
            },
        });

        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgb(135, 168, 206)',
            progressColor: 'rgb(43, 83, 131)',
            media: videoElement,
            autoScroll: true,
            minPxPerSec: 100,
            plugins: [regions, timeline],
        });

        wavesurfer.on('decode', () => {
            regions.clearRegions();
            activeRegion = regions.addRegion({
                start: data.initial_start,
                end: data.initial_end,
                color: 'rgba(246, 173, 85, 0.25)',
                drag: true,
                resize: true,
            });
            updateTimestampInputs();
        });

        regions.on('region-updated', (region) => {
            activeRegion = region;
            updateTimestampInputs();
        });

        // --- 核心修正区域 ---
        let isPlayingRegion = false;

        regions.on('region-out', (region) => {
            // 当播放光标离开区域时，如果是由“播放选区”功能启动的，则暂停
            if (isPlayingRegion) {
                wavesurfer.pause();
            }
        });
        
        // ★★★ 新增：监听 'pause' 事件以同步状态 ★★★
        // 无论因何原因暂停（手动或代码触发），都重置 isPlayingRegion 标志。
        // 这是解决“手动暂停后点击播放无效”问题的关键。
        wavesurfer.on('pause', () => {
            isPlayingRegion = false;
        });

        wavesurfer.on('finish', () => {
            isPlayingRegion = false;
        });
        
        // ★★★ 修正：“播放选区”按钮的逻辑 ★★★
        playSelectionBtn.onclick = async () => {
            if (activeRegion) {
                // 标记我们意图播放的是选区
                isPlayingRegion = true;
                try {
                    // 1. 手动定位到选区开始位置
                    const startProgress = activeRegion.start / wavesurfer.getDuration();
                    wavesurfer.seekTo(startProgress);

                    // 2. 调用 play() 并用 try/catch 处理浏览器自动播放限制
                    await wavesurfer.play();

                } catch (error) {
                    // 如果播放失败，在这里捕获错误并重置标志
                    isPlayingRegion = false; 
                    console.error('音频播放失败:', error);
                    alert(`浏览器阻止了自动播放。\n错误: ${error.name}\n请尝试先点击视频播放器下方的播放按钮与页面进行交互，然后再使用“播放选区”功能。`);
                }
            }
        };
    }

    function updateTimestampInputs() {
        if (activeRegion) {
            startTimeInput.value = activeRegion.start.toFixed(3);
            endTimeInput.value = activeRegion.end.toFixed(3);
        }
    }

    saveBtn.addEventListener('click', async () => {
        if (!activeRegion) {
            alert('没有可保存的选区。');
            return;
        }
        const finalStartTime = parseFloat(startTimeInput.value);
        const finalEndTime = parseFloat(endTimeInput.value);

        statusDisplay.textContent = '正在保存校对结果...';
        statusDisplay.style.backgroundColor = '#e2e8f0';

        try {
            const response = await fetch(`${API_BASE_URL}/save_correction`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start_time: finalStartTime, end_time: finalEndTime }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '保存失败');
            statusDisplay.textContent = `任务完成！已保存到: ${data.output_path}`;
            statusDisplay.style.backgroundColor = '#c6f6d5';
            correctorUI.classList.add('hidden');
        } catch (error) {
            statusDisplay.textContent = `保存错误: ${error.message}`;
            statusDisplay.style.backgroundColor = '#fed7d7';
        }
    });
});