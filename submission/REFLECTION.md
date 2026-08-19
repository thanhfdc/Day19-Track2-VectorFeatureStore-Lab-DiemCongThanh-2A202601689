# Reflection — Lab 19

**Tên:** Diêm Công Thành
**Cohort:** 3A
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên golden set 50 queries, hybrid thắng trung bình: BM25 77.8%, semantic
73.2%, hybrid 78.6%. Với `exact`, BM25 và hybrid gần ngang nhau vì query chứa
đúng thuật ngữ trong corpus. Với `mixed`, hybrid thắng rõ nhất vì BM25 bắt được
từ khóa còn vector bắt được ý nghĩa diễn đạt lại. Với `paraphrase`, semantic
không thắng như kỳ vọng vì lite path dùng `BAAI/bge-small-en-v1.5`, model thiên
về tiếng Anh nên yếu trên câu hỏi tiếng Việt.

Em không dùng hybrid khi query là exact ID, mã lỗi, tên hàm, log, hoặc thuật
ngữ pháp lý cần khớp chính xác; BM25 lúc đó nhanh và dễ giải thích hơn. Pure
vector hợp hơn khi người dùng hỏi mở, diễn đạt tự nhiên, và keyword trong corpus
khác nhiều.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid vẫn thắng dù embedding lite không mạnh cho tiếng Việt; RRF giúp hệ thống
ổn định hơn trên nhiều kiểu query.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _none_
