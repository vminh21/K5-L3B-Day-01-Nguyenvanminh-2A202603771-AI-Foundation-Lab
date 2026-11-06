# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603771**
- **Họ và tên:Nguyễn Văn Minh**
- **SEED** (in ở ô đầu notebook):1879

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

2. Kết quả chính

Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.

Mô hình	F1 (test)	F1 (SUB 40)	p95 ms/tin	USD / 1k tin
1. BoW + LogReg (tự code)	0.886	0.824	0.218	0.0
2. BoW + MLP (deep learning)	0.939	0.919	0.296	0.0
3a. Encoder (MiniLM) + kNN	0.918	0.974	230.952	0.0
3b. Encoder (MiniLM) + LogReg	0.962	0.974	35.785	0.0
3c. Encoder (MiniLM) + MLP head	0.968	0.974	49.239	0.0
4a. LLM nhỏ zero-shot	1.000	1.000	2298.909	0.0
4b. LLM lớn zero-shot	0.950	0.950	2367.589	0.0
4c. LLM nhỏ few-shot	0.974	0.974	2354.527	0.0

(Bảng đầy đủ các chỉ số: BoW+LogReg đạt Precision 0.987, Recall 0.804; BoW+MLP đạt Precision 0.966, Recall 0.913; Encoder+LogReg đạt Precision 0.957, Recall 0.967).

3. Dự đoán và thực tế

Chọn 3 dự đoán predict(...) lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.

1. Dự đoán kiểm thử 50 case khó (hard_eval)
Bạn đoán gì: Accuracy của các model giảm còn khoảng 65% – 75%. Nhóm khó nhất với BoW là Né từ khoá và Tưởng spam nhưng không phải.
Ra gì: BoW tụt dốc sâu hơn dự đoán rất nhiều: BoW + LogReg chỉ đạt 52% accuracy (F1 rơi xuống 0.333), BoW + MLP chỉ đạt 56% (F1 0.450). Đặc biệt ở nhóm "Tưởng spam nhưng không phải", BoW lại đoán đúng 100%. Nhóm khiến BoW sai nhiều nhất thực tế là "Spam giọng tự nhiên" (đúng 10%) và "Thao túng / lừa đảo" (đúng 12%).
Vì sao lệch: Nghĩ rằng các từ nhạy cảm như free, win sẽ làm BoW chặn nhầm tin ham, nhưng các tin nhắn bạn bè có nhiều từ sinh hoạt thường ngày kéo điểm về ham rất mạnh. Ngược lại, các tin lừa đảo tâm lý dùng giọng tự nhiên không hề có từ khóa spam nào, khiến BoW bị mù hoàn toàn và đoán nhầm 90% thành ham.
2. Dự đoán bỏ kích hoạt ReLU (dl_relu)
Bạn đoán gì: Nếu bỏ ReLU, mạng 2 lớp Linear liên tiếp sẽ suy biến thành mô hình tuyến tính đơn giản như Logistic Regression, F1 sẽ giảm về mức tương đương Phần 1 (~0.886).
Ra gì: F1 thực tế đạt tận 0.938 (vẫn xấp xỉ mức 0.939 của mạng có ReLU và vượt trội hơn mức 0.886 của LogReg thuần).
Vì sao lệch: Về toán học hai ma trận nhân liên tiếp đúng là tương đương một ma trận, nhưng kiến trúc đi qua lớp ẩn 64 chiều (Linear 2000 
→
→ 64 
→
→ 1) tạo ra hiệu ứng nén đặc trưng (low-rank factorization) giúp lọc nhiễu, cộng với việc dùng bộ tối ưu Adam trong PyTorch giúp hội tụ vào nghiệm tốt hơn nhiều so với thuật toán Gradient Descent cơ bản ở Phần 1.
3. Dự đoán tin nhắn bẫy ngụy trang (emb_tricky)
Bạn đoán gì: Mô hình Encoder + LogReg sẽ vượt trội hơn BoW trên các tin lắt léo vì hiểu được ngữ cảnh toàn câu.
Ra gì: Cả 2 mô hình đúng ngang nhau: cùng đúng 5/6 tin và cùng đoán sai đúng câu số 6: "Last chance! Your account rewards expire tonight, confirm details to keep them".
Vì sao lệch: Kỳ vọng Embedding là chiếc đũa thần giải quyết mọi bẫy từ ngữ, nhưng câu tin nhắn trên được ngụy trang hoàn toàn giống một tin nhắn thông báo dịch vụ bình thường (transactional). Khi không có bối cảnh bên ngoài (người gửi có uy tín không, người nhận có tài khoản đó không) thì ngay cả mô hình ngữ nghĩa sâu cũng bị đánh lừa.
4. Phân tích 50 case khó

Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.

Model tụt nhiều nhất:

Nhóm Bag-of-Words tụt nghiêm trọng nhất: BoW + LogReg tụt từ F1 SUB 0.824 xuống 0.333 (acc: 52%); BoW + MLP tụt từ 0.919 xuống 0.450 (acc: 56%).
Tụt nặng nhất ở nhóm: "Spam giọng tự nhiên" (acc: 10%) và "Thao túng / lừa đảo" (acc: 12% ở LogReg, 38% ở MLP).

Hai case cụ thể và cơ chế khiến model sai:

Case #21 (Nhóm Thao túng / lừa đảo):
Nội dung: "Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x" (Nhãn thật: spam).
Model sai: BoW + LogReg và BoW + MLP đoán thành ham.
Cơ chế sai: Kẻ lừa đảo đóng vai người thân trong tình huống khẩn cấp. Câu chứa toàn từ vựng sinh hoạt gia đình (Mum, phone, friend, explain, later), hoàn toàn vắng bóng các từ khóa spam quảng cáo (win, free, prize, claim). Mô hình BoW chỉ đếm từ độc lập nên các từ trên kéo điểm phạt về phía ham, khiến mô hình bỏ lọt tin nhắn lừa đảo.
Case #29 (Nhóm Né từ khoá):
Nội dung: "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m" (Nhãn thật: spam).
Model sai: BoW + LogReg và BoW + MLP đoán thành ham.
Cơ chế sai: Kẻ tấn công dùng kỹ thuật leetspeak thay chữ bằng số (0 thay cho o, 4 thay cho a, 1 thay cho i). Khi tách token, bộ tách từ tạo ra các chuỗi lạ như c0ngr4tulat10ns, cl4im không hề nằm trong từ điển 2.000 từ của tập train. Vector BoW bị biến thành toàn số 0 (Out-Of-Vocabulary), mô hình không tìm thấy bất kỳ tín hiệu spam nào nên mặc định xếp vào ham.
5. Thử thách cá nhân

Đề theo SEED của bạn, cách làm, kết quả, giải thích.

Đề theo SEED (SEED = 1879): Nhận đề #0:

"Thử vocab_size = 100, 2000, 5000 cho BoW + LogReg. F1 đổi thế nào? Giải thích vì sao."

Cách làm:
Lần lượt gọi hàm build_vocab(texts_tr, max_size=size) với kích thước 100, 2000 và 5000 từ.
Vector hóa tập train và tập test tương ứng, huấn luyện Logistic Regression (lr=0.5, epochs=300).
Đo các chỉ số Precision, Recall và F1 trên tập test tại ngưỡng quyết định 0.5.
Kết quả:
Vocab 100 từ: F1 = 0.795 (Precision = 0.969, Recall = 0.674)
Vocab 2000 từ: F1 = 0.886 (Precision = 0.987, Recall = 0.804)
Vocab 5000 từ: F1 = 0.886 (Precision = 0.987, Recall = 0.804)
Giải thích:
Khi giảm xuống 100 từ: F1 giảm mạnh từ 0.886 xuống 0.795 do Recall tụt nghiêm trọng (từ 0.804 xuống 0.674, bỏ sót hơn 32% lượng spam). Lý do là 100 từ phổ biến nhất đa phần là các từ ngữ pháp chung chung (i, to, you, the, in...), bỏ lọt hầu hết các từ khóa nhận diện spam quan trọng.
Khi tăng từ 2000 lên 5000 từ: Chỉ số giữ nguyên tuyệt đối không đổi (F1 = 0.886). Lý do là tập huấn luyện chỉ có khoảng hơn 2.000 từ xuất hiện đủ thường xuyên. Các từ từ vị trí 2001 trở đi là các từ cực hiếm (chỉ xuất hiện 1 lần do viết sai chính tả, ký tự ngẫu nhiên), không xuất hiện ở tập test và có trọng số học được xấp xỉ bằng 0 nên không làm xê dịch mặt phẳng phân chia quyết định. Ngưỡng 2.000 từ đã đạt điểm bão hòa tối ưu.
6. Phương án cho sản phẩm thật

Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.

Phương án lựa chọn: Chọn mô hình 3b. Encoder (MiniLM) + LogReg (hoặc kiến trúc Router lai lấy 3b làm lõi).
Dẫn chứng số liệu thực nghiệm:
Chất lượng: Đạt F1 test rất cao 0.962 (và F1 SUB đạt 0.974), vượt xa BoW (0.886) và bắt tốt các dạng tin nhắn ngụy trang.
Độ trễ: Đạt chuẩn SLA với p95 = 35.785 ms/tin (nằm an toàn dưới ngưỡng yêu cầu 50 ms).
Chi phí: 0 USD chi phí API, hoàn toàn tự host trên phần cứng CPU/GPU nội bộ với thông lượng xử lý hàng nghìn tin/giây.
(Loại trừ các phương án khác: BoW quá kém trên case khó ngoài đời; kNN có p95 = 230.95 ms vi phạm SLA; LLM có p95 > 2.200 ms vi phạm nghiêm trọng SLA và chi phí cho 10 triệu tin/ngày cực kỳ tốn kém).
Chấp nhận đánh đổi: Chấp nhận độ trễ xử lý lâu hơn BoW (35.8 ms so với 0.2 ms) và tốn tài nguyên máy chủ để chạy mô hình sinh vector embedding, đổi lại hệ thống có độ chính xác cao và khả năng chống chịu lừa đảo bền vững.
Mở rộng kiến trúc thực tế (Router 2 tầng):
Tầng 1 (Lọc nhanh - Encoder + LogReg): Xử lý 100% tin nhắn. Nếu xác suất 
𝑝
<
0.2
p<0.2 (chắc chắn ham) hoặc 
𝑝
>
0.8
p>0.8 (chắc chắn spam), chốt kết quả ngay trong vòng ~35 ms (giải quyết 95% lưu lượng).
Tầng 2 (LLM phân tích sâu): Khoảng 5% tin nhắn rơi vào vùng phân vân (
0.2
≤
𝑝
≤
0.8
0.2≤p≤0.8) hoặc chứa dấu hiệu tấn công đối kháng sẽ chuyển tiếp cho LLM nhỏ (kèm prompt phòng thủ) xử lý ngữ nghĩa chuyên sâu. Cách này vừa đảm bảo SLA p95 toàn hệ thống, vừa tiết kiệm 95% chi phí API.
7. AI log và điều đã học

Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.

Bảng AI Log:
Lần dùng	Bạn hỏi AI gì (prompt)	AI trả lời / viết gì	Bạn đã sửa / kiểm chứng gì
Phần 4 (LLM)	"Cú pháp gọi OpenAI client với provider_kwargs, cách viết prompt zero-shot/few-shot và biểu thức regex parse nhãn để không bắt nhầm các cụm từ như 'not spam'?"	Cung cấp template gọi client.chat.completions.create, prompt phân loại ngắn gọn và đoạn regex bắt từ khóa spam/ham.	Tự cấu hình provider phù hợp với API trong lab, bổ sung logic loại trừ tiền tố phủ định (r'(?<!not )\bspam\b') để tránh false positive, chạy test mock đạt 100%.
Điều hiểu ra hôm nay:
Biểu diễn dữ liệu (Representation) quyết định chất lượng: Bước nhảy vọt về F1 đến từ việc chuyển đổi từ Bag-of-Words rời rạc sang Dense Embedding ngữ nghĩa, chứ không đến từ việc cố xếp thêm nhiều lớp nơ-ron sâu (MLP) trên một biểu diễn nghèo nàn.
Kỹ thuật AI trong sản phẩm là bài toán đánh đổi: Không có mô hình nào tốt nhất ở mọi mặt; thiết kế hệ thống thực tế là tìm điểm cân bằng hài hòa giữa chất lượng (F1), độ trễ (Latency p95) và chi phí vận hành (Cost).
Cảnh giác với số liệu kiểm thử: Điểm F1 cao trên tập test phân phối chuẩn (I.I.D) dễ tạo cảm giác an tâm giả tạo; chỉ các bộ test đối kháng (stress test 50 case khó) mới bộc lộ chính xác điểm mù của mô hình trước các thủ đoạn lừa đảo ngoài đời thực.
