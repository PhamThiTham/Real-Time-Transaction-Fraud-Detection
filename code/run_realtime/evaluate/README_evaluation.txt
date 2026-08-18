1. chạy monitor_resources.py

Lưu ý: chạy transaction_producer.py trước khi chạy resource_monitor.py
Tốt nhất là bắt đầu monitor ngay trước hoặc cùng lúc với producer.

cd /d D:\ThucTap_VinSmartFuture\run_realtime\evaluate

Muốn test khoảng 5 phút, chạy:
python monitor_resources.py --duration 300 --interval 1 
Trong đó:
--duration 300 = theo dõi 300 giây = 5 phút
--interval 1 = lấy mẫu mỗi 1 giây
