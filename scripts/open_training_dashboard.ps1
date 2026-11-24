param(
    [ValidatePattern('^[A-Za-z0-9.-]+$')][string]$ServerHost = 'connect.nmb2.seetacloud.com',
    [ValidateRange(1, 65535)][int]$SshPort = 28306,
    [ValidateRange(1, 65535)][int]$LocalPort = 8866,
    [ValidateRange(1, 65535)][int]$RemotePort = 8866
)
$ErrorActionPreference = 'Stop'
$dashboardUrl = "http://127.0.0.1:$LocalPort"
function Test-Dashboard {
    try {
        $response = Invoke-RestMethod "$dashboardUrl/api/overview" -TimeoutSec 2
        return ($response.max_concurrent -eq 2 -and $null -ne $response.resources)
    } catch { return $false }
}
if (-not (Test-Dashboard)) {
    $sshExecutable = (Get-Command ssh -ErrorAction Stop).Source
    $sshArguments = @('-N', '-L', "${LocalPort}:127.0.0.1:${RemotePort}", '-p', "$SshPort",
        '-o', 'ExitOnForwardFailure=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', "root@$ServerHost")
    # 此窗口用于用户交互输入SSH密码；密码不进入脚本、参数或面板数据库。
    $tunnelProcess = Start-Process -FilePath $sshExecutable -ArgumentList $sshArguments -WindowStyle Normal -PassThru
    Write-Host '请在新打开的 SSH 窗口输入服务器密码，并保持该窗口打开。'
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if (Test-Dashboard) { break }
        if ($tunnelProcess.HasExited) { throw 'SSH 转发已退出，请检查端口或登录信息。' }
        Start-Sleep -Seconds 1
    }
    if (-not (Test-Dashboard)) { throw '等待面板超时，请检查服务器面板服务。' }
}
Start-Process $dashboardUrl
Write-Host "训练面板已打开：$dashboardUrl"
