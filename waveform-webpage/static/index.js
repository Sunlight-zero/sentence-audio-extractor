document.addEventListener('DOMContentLoaded', () => {
    console.log("页面加载完成，script.js 开始执行。");

    // --- 获取元素 ---
    const form = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    const statusContainer = document.getElementById('status-container');
    const statusMessage = document.getElementById('status-message');
    const resultContainer = document.getElementById('result-container');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const proofreadLinkBtn = document.getElementById('proofread-link-btn');
    
    const sentenceTextarea = document.getElementById('sentence');
    const sentenceFileInput = document.getElementById('sentenceFile');

    // --- 新增元素获取 ---
    const getAnkiBtn = document.getElementById('get-anki-btn');
    const deckNameInput = document.getElementById('deck-name');
    const videoFilesInput = document.getElementById('videoFiles'); // 获取新的多文件输入框

    let pollInterval;

    // --- 新增功能: 从 Anki 获取句子 ---
    if (getAnkiBtn) {
        getAnkiBtn.addEventListener('click', async () => {
            const deckName = deckNameInput.value.trim() || 'luna temporary';
            statusContainer.style.display = 'block';
            statusMessage.textContent = `正在从 Anki 牌组 '${deckName}' 获取句子...`;
            getAnkiBtn.disabled = true;

            try {
                const response = await fetch(`/api/anki/sentences?deck=${encodeURIComponent(deckName)}`);
                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || `服务器错误，状态码: ${response.status}`);
                }
                
                sentenceTextarea.value = data.sentences;
                statusMessage.textContent = data.message || '句子获取成功！';

            } catch (error) {
                console.error('从 Anki 获取句子时出错:', error);
                statusMessage.textContent = `错误: ${error.message}`;
            } finally {
                getAnkiBtn.disabled = false;
            }
        });
    }

    // --- 监听文件选择框的变化事件 (无变动) ---
    if (sentenceFileInput && sentenceTextarea) {
        sentenceFileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                sentenceTextarea.value = e.target.result;
            };
            reader.onerror = (e) => {
                console.error("读取文件时出错:", e);
                alert("读取文件时发生错误。");
            };
            reader.readAsText(file, 'UTF-8');
        });
    }

    // --- 核心事件监听 (修改以支持多文件) ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log("表单提交事件被触发。");

        submitBtn.disabled = true;
        submitBtn.textContent = '正在上传...';
        resultContainer.style.display = 'none';
        statusContainer.style.display = 'block';
        statusMessage.textContent = '文件上传中，请稍候...';

        // --- 修改: 直接使用 FormData，它能自动处理多文件 ---
        const formData = new FormData(form);
        // 验证是否选择了文件
        if (!formData.has('videoFiles') || !videoFilesInput.files.length) {
            statusMessage.textContent = '错误: 请至少选择一个视频文件。';
            submitBtn.disabled = false;
            submitBtn.textContent = '开始处理';
            return;
        }
        console.log("已创建 FormData，包含多个文件。");

        try {
            console.log("准备发送 fetch 请求到 /process...");
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });
            
            console.log("收到来自服务器的响应:", response);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `服务器错误，状态码: ${response.status}`);
            }
            
            if (data.status === 'processing' && data.task_id) {
                submitBtn.textContent = '处理中...';
                statusMessage.textContent = '文件上传成功，后台正在处理...';
                pollStatus(data.task_id);
            } else {
                throw new Error(data.error || '未能从服务器获取有效的任务ID。');
            }

        } catch (error) {
            console.error('处理请求时出错:', error);
            statusMessage.textContent = `错误: ${error.message}`;
            submitBtn.disabled = false;
            submitBtn.textContent = '开始处理';
        }
    });

    function pollStatus(taskId) {
        console.log(`开始轮询任务状态，任务ID: ${taskId}`);
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/task_status/${taskId}`);
                if (!res.ok) {
                    console.warn(`无法连接服务器检查状态，状态码: ${res.status}。将在3秒后重试。`);
                    return;
                }
                const data = await res.json();
                statusMessage.textContent = data.message || data.error || `当前状态: ${data.status}`;

                if (data.status === 'completed') {
                    console.log("任务完成！", data);
                    clearInterval(pollInterval);
                    
                    if (data.result_json_url) {
                        statusContainer.style.display = 'none';
                        resultContainer.style.display = 'block';
                        
                        downloadJsonBtn.href = data.result_json_url;
                        downloadJsonBtn.download = data.result_json_url.split('/').pop();

                        // --- 修改: 不再传递 video_url，因为它现在是动态的 ---
                        const proofreadUrl = `/proofread?json_url=${encodeURIComponent(data.result_json_url)}`;
                        proofreadLinkBtn.href = proofreadUrl;
                    } else {
                        // 处理没有匹配结果的情况
                        statusContainer.style.display = 'block'; // 保持状态信息可见
                        resultContainer.style.display = 'none';
                    }

                    submitBtn.disabled = false;
                    submitBtn.textContent = '开始处理';

                } else if (data.status === 'error') {
                    console.error("后台任务处理失败:", data.error);
                    clearInterval(pollInterval);
                    statusMessage.textContent = `处理失败: ${data.error}`;
                    submitBtn.disabled = false;
                    submitBtn.textContent = '重新处理';
                }
            } catch (error) {
                console.error('轮询状态时发生严重错误:', error);
                statusMessage.textContent = `轮询错误: ${error.message}。已停止轮询。`;
                clearInterval(pollInterval);
                submitBtn.disabled = false;
                submitBtn.textContent = '重新处理';
            }
        }, 3000);
    }
});
