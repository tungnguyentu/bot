import subprocess
import time
import sys
import os
import logging
import traceback
import signal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("watchdog.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Watchdog:
    def __init__(self, script_path, max_restarts=5, restart_cooldown=300):
        """
        Initialize the watchdog.
        
        Args:
            script_path: Path to the script to monitor and restart
            max_restarts: Maximum number of restarts within cooldown period
            restart_cooldown: Cooldown period in seconds
        """
        self.script_path = script_path
        self.max_restarts = max_restarts
        self.restart_cooldown = restart_cooldown
        self.restarts = []
        self.process = None
        self.running = True
        
    def start(self):
        """Start the monitored script and watchdog loop."""
        logger.info(f"Watchdog starting to monitor: {self.script_path}")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        while self.running:
            try:
                # Start the process
                start_time = datetime.now()
                logger.info(f"Starting process at {start_time}")
                
                # Execute the script
                self.process = subprocess.Popen(
                    [sys.executable, self.script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Monitor the process output
                for line in self.process.stdout:
                    print(line, end='')  # Echo to console
                
                # Wait for process to complete
                self.process.wait()
                end_time = datetime.now()
                run_time = (end_time - start_time).total_seconds()
                
                # Check exit code
                exit_code = self.process.returncode
                logger.info(f"Process exited with code {exit_code} after running for {run_time:.1f} seconds")
                
                # Check if we should restart
                if not self.running:
                    logger.info("Watchdog shutting down, not restarting process")
                    break
                    
                if not self.should_restart():
                    logger.warning("Too many restarts in cooldown period. Watchdog shutting down.")
                    break
                
                # Record restart
                self.restarts.append(datetime.now())
                logger.info(f"Restarting process in 5 seconds...")
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in watchdog: {e}")
                logger.error(traceback.format_exc())
                if self.running:
                    time.sleep(10)  # Delay before retry on error
                else:
                    break
            
    def should_restart(self):
        """
        Check if we should restart based on restart history.
        
        Returns:
            bool: True if restart is allowed, False otherwise
        """
        # Remove restarts older than cooldown period
        now = datetime.now()
        self.restarts = [t for t in self.restarts if (now - t).total_seconds() < self.restart_cooldown]
        
        # Check if we've exceeded max restarts
        return len(self.restarts) < self.max_restarts
    
    def handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down")
        self.running = False
        
        if self.process:
            logger.info("Terminating monitored process")
            self.process.terminate()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watchdog.py <script_path>")
        sys.exit(1)
        
    script_path = sys.argv[1]
    watchdog = Watchdog(script_path)
    watchdog.start()
