# PowerShell script to watch grading_log.txt and auto-push changes to GitHub
$path = "AI_Grading_Project/grading_log.txt"
$lastHash = ""

while ($true) {
    if (Test-Path $path) {
        $currentHash = Get-FileHash $path | Select-Object -ExpandProperty Hash
        if ($currentHash -ne $lastHash) {
            git add $path
            git commit -m "Auto-update grading_log.txt"
            git push origin master
            $lastHash = $currentHash
        }
    }
    Start-Sleep -Seconds 600  # Sleep for 10 minutes
}
