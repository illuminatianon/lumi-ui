"""Filesystem management for Lumi UI configuration."""

import os
import stat
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ConfigFileSystem:
    """Manages the ~/.lumi directory structure and file permissions."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize filesystem manager.
        
        Args:
            config_dir: Override default config directory (for testing)
        """
        if config_dir:
            self.lumi_dir = Path(config_dir)
        else:
            self.lumi_dir = Path.home() / ".lumi"
        
        self.config_file = self.lumi_dir / "config.hjson"
        self.state_file = self.lumi_dir / "state.hjson"
        self.cache_dir = self.lumi_dir / "cache"
        self.tmp_dir = self.lumi_dir / "tmp"
        self.logs_dir = self.lumi_dir / "logs"
    
    def ensure_directory_structure(self) -> bool:
        """Create required directory structure with proper permissions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create main .lumi directory
            self.lumi_dir.mkdir(mode=0o700, exist_ok=True)
            logger.info(f"Created/verified main directory: {self.lumi_dir}")
            
            # Create required subdirectories
            required_dirs = [self.cache_dir, self.tmp_dir]
            for directory in required_dirs:
                directory.mkdir(mode=0o700, exist_ok=True)
                logger.info(f"Created/verified directory: {directory}")
            
            # Set proper permissions on existing directories
            self._set_directory_permissions(self.lumi_dir, 0o700)
            for directory in required_dirs:
                self._set_directory_permissions(directory, 0o700)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directory structure: {e}")
            return False
    
    def ensure_logs_directory(self) -> bool:
        """Create logs directory if logging is enabled.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logs_dir.mkdir(mode=0o700, exist_ok=True)
            self._set_directory_permissions(self.logs_dir, 0o700)
            logger.info(f"Created/verified logs directory: {self.logs_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create logs directory: {e}")
            return False
    
    def ensure_config_file_permissions(self) -> bool:
        """Ensure configuration files have secure permissions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            files_to_secure = [self.config_file, self.state_file]
            for file_path in files_to_secure:
                if file_path.exists():
                    self._set_file_permissions(file_path, 0o600)
                    logger.info(f"Set secure permissions on: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to set file permissions: {e}")
            return False
    
    def _set_directory_permissions(self, directory: Path, mode: int) -> None:
        """Set permissions on a directory.
        
        Args:
            directory: Directory path
            mode: Permission mode (octal)
        """
        try:
            directory.chmod(mode)
        except Exception as e:
            logger.warning(f"Could not set permissions on {directory}: {e}")
    
    def _set_file_permissions(self, file_path: Path, mode: int) -> None:
        """Set permissions on a file.
        
        Args:
            file_path: File path
            mode: Permission mode (octal)
        """
        try:
            file_path.chmod(mode)
        except Exception as e:
            logger.warning(f"Could not set permissions on {file_path}: {e}")
    
    def cleanup_tmp_directory(self, max_age_hours: int = 24) -> bool:
        """Clean up old files in tmp directory.
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            if not self.tmp_dir.exists():
                return True
            
            cleaned_count = 0
            for file_path in self.tmp_dir.rglob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Could not delete {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old files from tmp directory")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup tmp directory: {e}")
            return False
    
    def get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes.
        
        Args:
            directory: Directory to measure
            
        Returns:
            Size in bytes, 0 if error
        """
        try:
            total_size = 0
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            logger.warning(f"Could not calculate size of {directory}: {e}")
            return 0
    
    def check_disk_space(self, min_free_mb: int = 100) -> bool:
        """Check if there's sufficient disk space.
        
        Args:
            min_free_mb: Minimum free space required in MB
            
        Returns:
            True if sufficient space available
        """
        try:
            statvfs = os.statvfs(self.lumi_dir.parent)
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            free_mb = free_bytes / (1024 * 1024)
            return free_mb >= min_free_mb
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Assume OK if we can't check
    
    def create_backup(self, source_file: Path, backup_suffix: str = ".backup") -> bool:
        """Create a backup of a configuration file.
        
        Args:
            source_file: File to backup
            backup_suffix: Suffix for backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not source_file.exists():
                return True
            
            backup_file = source_file.with_suffix(source_file.suffix + backup_suffix)
            backup_file.write_bytes(source_file.read_bytes())
            self._set_file_permissions(backup_file, 0o600)
            
            logger.info(f"Created backup: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup of {source_file}: {e}")
            return False
