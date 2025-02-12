import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from .base import BaseLogger, StepLog, WorkflowLog

class JsonLogger(BaseLogger):
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Separate directories for workflows and steps
        self.workflow_dir = self.log_dir / "workflows"
        self.workflow_dir.mkdir(exist_ok=True)
        
        self.step_dir = self.log_dir / "steps"
        self.step_dir.mkdir(exist_ok=True)
    
    def _serialize(self, obj: Any) -> Any:
        """Simple serialization of objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    async def log_step(self, step_log: StepLog) -> None:
        """Log individual step"""
        step_file = self.step_dir / f"{step_log.step_id}.json"
        with open(step_file, "w") as f:
            json.dump(self._serialize_step(step_log), f, indent=2)
    
    async def log_workflow(self, workflow_log: WorkflowLog) -> None:
        """Log workflow with step references"""
        workflow_file = self.workflow_dir / f"{workflow_log.workflow_id}.json"
        with open(workflow_file, "w") as f:
            json.dump(self._serialize_workflow(workflow_log), f, indent=2)
    
    def get_workflow_logs(self, 
                         workflow_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[WorkflowLog]:
        """Get workflow logs with optional filtering"""
        if workflow_id:
            workflow_files = [self.workflow_dir / f"{workflow_id}.json"]
        else:
            workflow_files = list(self.workflow_dir.glob("*.json"))
        
        workflows = []
        for wf_file in workflow_files:
            if wf_file.exists():
                try:
                    with open(wf_file) as f:
                        data = json.load(f)
                        workflows.append(data)
                except Exception as e:
                    print(f"Error reading workflow log {wf_file}: {e}")
                    continue
        
        return workflows 