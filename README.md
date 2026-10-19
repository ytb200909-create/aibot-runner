# aibot-runner

Telegram AI Controller chạy 24/7 trên GitHub Actions (repo public →
số phút chạy không giới hạn, không cần thẻ tín dụng).

- Bot dùng long-polling (chỉ cần truy cập internet ra ngoài), không cần mở port.
- Mỗi giờ workflow tự khởi động lại bot (giới hạn 1 job GitHub Actions tối đa 60 phút).
- Tin nhắn gửi trong lúc bot đang khởi động lại **không bị mất** — Telegram
  giữ update tối đa 24h và bot sẽ nhận được khi online trở lại.
- Mọi cấu hình (token, API key) nằm trong GitHub Secret `BOT_CONFIG`,
  **không** nằm trong code repo.

## Lệnh quản trị (GitHub CLI)

```bash
gh run list -R ytb200909-create/aibot-runner --limit 5   # xem các lần chạy
gh workflow run bot.yml -R ytb200909-create/aibot-runner # chạy ngay
gh secret set BOT_CONFIG -R ytb200909-create/aibot-runner < config.json
```

## Lưu ý

- GitHub tự tắt cron workflow nếu repo **60 ngày không có commit**.
  Cứ ~1-2 tháng cần 1 commit nhỏ (hoặc chạy `gh workflow run` thủ công).
- Bot AI dùng provider cloud (Groq/OpenAI...). Lệnh AI "local" (LM Studio
  trên máy nhà) không khả dụng trên runner.
