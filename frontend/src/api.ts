/**
 * DSPy Prompt Optimizer - API Client
 */

const API_BASE = '/api';

export interface ReActStep {
  id: string;
  name: string;
  tool: string;
  status: 'pending' | 'running' | 'success' | 'error';
  thought?: string;
  action?: string;
  observation?: string;
  duration_ms?: number;
  error?: string;
  timestamp?: string;
}

export interface TaskAnalysis {
  task_type: string;
  domain: string;
  input_roles: string[];
  output_roles: string[];
  needs_retrieval: boolean;
  needs_chain_of_thought: boolean;
  complexity_level: string;
  safety_level: string;
}

export interface OrchestratorResult {
  artifact_version_id: string;
  compiled_program_id: string;
  signature_id: string;
  eval_results: {
    metric_name?: string;
    metric_value?: number;
    dev_accuracy?: number;
    test_accuracy?: number;
    iterations?: number;
    metric_history?: number[];
    real_dspy?: boolean;
  };
  task_analysis?: TaskAnalysis;
  program_code: string;
  deployment_package?: {
    path: string;
    instructions: string;
  };
  react_iterations: number;
  total_cost_usd?: number;
  optimizer_type?: string;
  quality_profile?: string;
  data_splits?: {
    train: number;
    dev: number;
    test: number;
  };
}

export interface OrchestratorRequest {
  business_task: string;
  target_lm: string;
  optimizer_lm: string;
  dataset: Array<{ input: string; output: string }>;
  quality_profile: string;
  optimizer_strategy: string;
  use_agent?: boolean;
}

export interface TestArtifactRequest {
  artifact_id: string;
  input_text: string;
  target_lm: string;
  program_code?: string;
}

interface StreamCallbacks {
  onStep: (step: ReActStep) => void;
  onComplete: (result: OrchestratorResult) => void;
  onError: (error: string) => void;
}

export interface OllamaStatus {
  available: boolean;
  models_count: number;
  base_url: string;
}

export const api = {
  /**
   * Get available models for a provider
   */
  async getModels(provider: string): Promise<string[]> {
    const response = await fetch(`${API_BASE}/models/${provider}`);
    if (!response.ok) {
      throw new Error(`Failed to get models: ${response.statusText}`);
    }
    const data = await response.json();
    return data.models || [];
  },

  /**
   * Check Ollama status
   */
  async getOllamaStatus(): Promise<OllamaStatus> {
    const response = await fetch(`${API_BASE}/ollama/status`);
    if (!response.ok) {
      return { available: false, models_count: 0, base_url: 'http://localhost:11434' };
    }
    return response.json();
  },

  /**
   * Stream DSPy orchestration with Server-Sent Events
   */
  streamOrchestrate(request: OrchestratorRequest, callbacks: StreamCallbacks): () => void {
    const controller = new AbortController();

    (async () => {
      try {
        const response = await fetch(`${API_BASE}/dspy/orchestrate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(request),
          signal: controller.signal,
        });

        if (!response.ok) {
          const error = await response.text();
          callbacks.onError(error || response.statusText);
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError('No response body');
          return;
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'step') {
                  callbacks.onStep(data.step || data.data);
                } else if (data.type === 'complete') {
                  callbacks.onComplete(data.data || data);
                } else if (data.type === 'error') {
                  callbacks.onError(data.error);
                }
              } catch (e) {
                console.error('Failed to parse SSE data:', e);
              }
            }
          }
        }
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') {
          return;
        }
        callbacks.onError(e instanceof Error ? e.message : 'Unknown error');
      }
    })();

    return () => controller.abort();
  },

  /**
   * Test an optimized artifact
   */
  async testArtifact(request: TestArtifactRequest): Promise<string> {
    const response = await fetch(`${API_BASE}/dspy/test`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error || response.statusText);
    }

    const data = await response.json();
    return data.output;
  },

  /**
   * List all artifacts
   */
  async listArtifacts(): Promise<OrchestratorResult[]> {
    const response = await fetch(`${API_BASE}/artifacts`);
    if (!response.ok) {
      throw new Error(`Failed to list artifacts: ${response.statusText}`);
    }
    const data = await response.json();
    return data.artifacts || [];
  },

  /**
   * Get a specific artifact
   */
  async getArtifact(artifactId: string): Promise<{ metadata: OrchestratorResult; program_code: string }> {
    const response = await fetch(`${API_BASE}/artifacts/${artifactId}`);
    if (!response.ok) {
      throw new Error(`Failed to get artifact: ${response.statusText}`);
    }
    return response.json();
  },

  // ==================== HuggingFace Dataset Catalog ====================

  /**
   * Search HuggingFace datasets
   */
  async searchHFDatasets(query: string, limit: number = 20): Promise<{
    results: Array<{
      id: string;
      cardData?: any;
      tags?: string[];
      downloads?: number;
      likes?: number;
    }>;
  }> {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    const response = await fetch(`${API_BASE}/datasets/catalog/hf/search?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Failed to search HF datasets: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Inspect HuggingFace dataset columns
   */
  async inspectHFDataset(request: {
    dataset_id: string;
    config_name?: string;
    split?: string;
  }): Promise<{
    columns: string[];
    suggested_input?: string | null;
    suggested_output?: string | null;
  }> {
    const response = await fetch(`${API_BASE}/datasets/catalog/hf/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new Error(`Failed to inspect HF dataset: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Import HuggingFace dataset
   */
  async importHFDataset(request: {
    dataset_id: string;
    config_name?: string;
    split?: string;
    input_key?: string;
    output_key?: string;
    max_items?: number;
  }): Promise<{
    name: string;
    description: string;
    items: Array<{ input: string; output: string }>;
    meta: any;
  }> {
    const response = await fetch(`${API_BASE}/datasets/catalog/hf/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new Error(`Failed to import HF dataset: ${response.statusText}`);
    }
    return response.json();
  },
};
