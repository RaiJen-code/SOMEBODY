#!/usr/bin/env python3
"""
SOMEBODY GUI - Graphical User Interface for the SOMEBODY system
"""
import sys
import os
import subprocess
import threading
import time
import argparse
from PyQt5.QtWidgets import QApplication

# Parse command line arguments
parser = argparse.ArgumentParser(description='SOMEBODY - Smart Assistant GUI')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--standalone', action='store_true', help='Run without starting backend server')
args = parser.parse_args()

# Deteksi direktori proyek dan atur path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Tambahkan root proyek ke PYTHONPATH

# Konfigurasi logging
import logging
logging_level = logging.DEBUG if args.debug else logging.INFO
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, "logs", f"somebody_gui_{time.strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modul server dan controller
try:
    from integration.server import start_server
    from integration.controller import SomebodyController
    from config.settings import API_HOST, API_PORT
except ImportError as e:
    logger.error(f"Tidak dapat mengimpor modul SOMEBODY: {str(e)}")
    sys.exit(1)

# Import main window
try:
    from ui.main_window import SomebodyMainWindow
except ImportError as e:
    logger.error(f"Tidak dapat mengimpor SomebodyMainWindow: {str(e)}")
    sys.exit(1)

# Fungsi untuk menjalankan backend API di thread terpisah
def run_backend_api():
    logger.info("Memulai backend API server...")
    try:
        # Initialize controller terlebih dahulu
        controller = SomebodyController()
        controller.initialize()
        controller.load_state()
        
        # Mulai server API
        start_server()
    except Exception as e:
        logger.error(f"Error saat menjalankan backend API: {str(e)}")

def main():
    logger.info("Memulai aplikasi SOMEBODY GUI...")
    logger.debug(f"Project root: {current_dir}")
    
    # Cek apakah server sudah berjalan, kecuali jika --standalone
    server_running = args.standalone
    
    if not server_running:
        import requests
        try:
            # Coba akses API endpoint untuk cek apakah server sudah berjalan
            logger.info(f"Memeriksa API di http://{API_HOST}:{API_PORT}/...")
            response = requests.get(f"http://{API_HOST}:{API_PORT}/", timeout=2)
            if response.status_code == 200:
                logger.info("Backend API sudah berjalan")
                server_running = True
            else:
                logger.warning(f"Backend API merespons tapi dengan status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.info("Backend API belum berjalan. Akan dimulai...")
        except Exception as e:
            logger.error(f"Error saat memeriksa status server: {str(e)}")
        
        # Jika server belum berjalan, mulai di thread terpisah
        if not server_running:
            logger.info("Memulai backend API di thread terpisah...")
            api_thread = threading.Thread(target=run_backend_api, daemon=True)
            api_thread.start()
            
            # Tunggu sebentar sampai API siap
            logger.info("Menunggu backend API siap...")
            for i in range(40):  # Tunggu maksimal 20 detik, cek setiap 0.5 detik
                try:
                    response = requests.get(f"http://{API_HOST}:{API_PORT}/", timeout=1)
                    if response.status_code == 200:
                        logger.info("Backend API siap")
                        break
                except:
                    pass
                time.sleep(0.5)
    
    # Jalankan aplikasi GUI
    try:
        logger.info("Memulai aplikasi GUI...")
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Gunakan style Fusion untuk tampilan konsisten
        
        # Muat stylesheet jika ada
        style_file = os.path.join(current_dir, "ui", "style.qss")
        logger.debug(f"Mencari stylesheet di: {style_file}")
        
        if os.path.exists(style_file):
            logger.info(f"Memuat stylesheet dari {style_file}")
            with open(style_file, "r") as f:
                style_content = f.read()
                app.setStyleSheet(style_content)
                logger.debug(f"Stylesheet dimuat: {len(style_content)} karakter")
        else:
            logger.warning(f"File stylesheet tidak ditemukan di {style_file}")
        
        # Inisialisasi jendela utama
        logger.info("Menginisialisasi jendela utama...")
        window = SomebodyMainWindow()
        window.show()
        
        logger.info("Aplikasi GUI siap")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Error saat menjalankan aplikasi GUI: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()