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
    
    // --- 诊断日志 ---
    if (!form) {
        console.error("错误：无法找到 ID 为 'upload-form' 的表单元素。请检查 index.html。");
        return;
    }
    if (!submitBtn) {
        console.error("错误：无法找到 ID 为 'submit-btn' 的按钮元素。");
        return;
    }
    console.log("成功获取到表单和按钮元素。", { form, submitBtn });

    let pollInterval;

    // --- 核心事件监听 ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // 阻止表单默认的提交刷新行为
        console.log("表单提交事件被触发。");

        // 禁用按钮并显示加载状态
        submitBtn.disabled = true;
        submitBtn.textContent = '正在上传...';
        resultContainer.style.display = 'none';
        statusContainer.style.display = 'block';
        statusMessage.textContent = '文件上传中，请稍候...';

        const formData = new FormData(form);
        console.log("已创建 FormData:", formData);

        try {
            console.log("准备发送 fetch 请求到 /process...");
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });
            
            console.log("收到来自服务器的响应:", response);

            const data = await response.json();

            if (!response.ok) {
                // 如果服务器返回错误 (如 400, 500), 抛出错误
                throw new Error(data.error || `服务器错误，状态码: ${response.status}`);
            }
            
            if (data.status === 'processing' && data.task_id) {
                submitBtn.textContent = '处理中...';
                statusMessage.textContent = '文件上传成功，后台正在处理...';
                // 开始轮询任务状态
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
                    // 不清除轮询，因为可能是暂时的网络问题
                    console.warn(`无法连接服务器检查状态，状态码: ${res.status}。将在3秒后重试。`);
                    return;
                }
                const data = await res.json();

                // 实时更新状态消息
                statusMessage.textContent = data.message || data.error || `当前状态: ${data.status}`;

                if (data.status === 'completed') {
                    console.log("任务完成！", data);
                    clearInterval(pollInterval);
                    statusContainer.style.display = 'none';
                    resultContainer.style.display = 'block';
                    
                    // 设置下载和跳转链接
                    downloadJsonBtn.href = data.result_json_url;
                    downloadJsonBtn.download = data.result_json_url.split('/').pop();

                    const proofreadUrl = `/proofread?video_url=${encodeURIComponent(data.video_url)}&json_url=${encodeURIComponent(data.result_json_url)}`;
                    proofreadLinkBtn.href = proofreadUrl;

                    // 恢复按钮状态
                    submitBtn.disabled = false;
                    submitBtn.textContent = '开始处理';

                } else if (data.status === 'error') {
                    console.error("后台任务处理失败:", data.error);
                    clearInterval(pollInterval);
                    statusMessage.textContent = `处理失败: ${data.error}`;
                    submitBtn.disabled = false;
                    submitBtn.textContent = '重新处理';
                }
                // 如果状态是 'processing'，则不执行任何操作，继续轮询
            } catch (error) {
                console.error('轮询状态时发生严重错误:', error);
                statusMessage.textContent = `轮询错误: ${error.message}。已停止轮询。`;
                clearInterval(pollInterval);
                submitBtn.disabled = false;
                submitBtn.textContent = '重新处理';
            }
        }, 3000); // 每3秒查询一次状态
    }
});
