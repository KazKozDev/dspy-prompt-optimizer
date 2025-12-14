import { useState, useEffect, useRef, useMemo } from 'react';
import { Settings, Play, Download, Copy, Check, Database, Cpu, AlertCircle, Sparkles, FlaskConical, Rocket, Search, X } from 'lucide-react';

// Custom DSPy Logo Component
const DSPyLogo = ({ className = "w-12 h-12" }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Blue rounded square background */}
    <rect x="5" y="5" width="90" height="90" rx="20" fill="#2563EB" />
    
    {/* White letter D */}
    <path 
      d="M28 22 L28 78 L52 78 C72 78 84 64 84 50 C84 36 72 22 52 22 L28 22 Z M40 34 L52 34 C64 34 72 40 72 50 C72 60 64 66 52 66 L40 66 L40 34 Z"
      fill="white"
    />
  </svg>
);
import { api, type OrchestratorResult, type ReActStep, type OllamaStatus } from './api';

// Provider configurations
const PROVIDERS = [
  { id: 'openai', label: 'OpenAI', requiresKey: true },
  { id: 'anthropic', label: 'Anthropic', requiresKey: true },
  { id: 'gemini', label: 'Google Gemini', requiresKey: true },
  { id: 'ollama', label: 'Ollama (Local)', requiresKey: false },
] as const;

// Quality profiles
const QUALITY_PROFILES = [
  { value: 'FAST_CHEAP', label: 'Fast', description: 'Quick prototyping, minimal iterations' },
  { value: 'BALANCED', label: 'Balanced', description: 'Best for most use cases' },
  { value: 'HIGH_QUALITY', label: 'Quality', description: 'Maximum optimization, slower' },
] as const;

// Optimizer strategies
const OPTIMIZER_STRATEGIES = [
  { value: 'auto', label: 'Auto', description: 'Agent picks best strategy' },
  { value: 'BootstrapFewShot', label: 'Bootstrap', description: 'Fast, works with 10-50 examples' },
  { value: 'MIPROv2', label: 'MIPRO v2', description: 'Best quality, needs 50+ examples' },
  { value: 'COPRO', label: 'COPRO', description: 'Instruction optimization' },
] as const;

interface ModelOption {
  value: string;
  label: string;
  provider: string;
}

function App() {
  // App loading state
  const [isAppLoading, setIsAppLoading] = useState(true);

  // Settings state
  const [showSettings, setShowSettings] = useState(false);
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({
    openai: '',
    anthropic: '',
    gemini: '',
  });

  // Model selection
  const [targetModels, setTargetModels] = useState<ModelOption[]>([]);
  const [optimizerModels, setOptimizerModels] = useState<ModelOption[]>([]);
  const [targetLM, setTargetLM] = useState('');
  const [optimizerLM, setOptimizerLM] = useState('');

  // Configuration
  const [qualityProfile, setQualityProfile] = useState('BALANCED');
  const [optimizerStrategy, setOptimizerStrategy] = useState('auto');
  const [useAgent, setUseAgent] = useState(true);

  // Input state
  const [businessTask, setBusinessTask] = useState('');
  const [datasetText, setDatasetText] = useState('');

  // Process state
  const [isRunning, setIsRunning] = useState(false);
  const [currentPhase, setCurrentPhase] = useState('');
  const [reactSteps, setReactSteps] = useState<ReActStep[]>([]);
  const [result, setResult] = useState<OrchestratorResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // UI state
  const [activeTab, setActiveTab] = useState<'overview' | 'code' | 'test' | 'deploy'>('overview');
  const [copied, setCopied] = useState(false);
  const [testInput, setTestInput] = useState('');
  const [testOutput, setTestOutput] = useState<string | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null);

  // HuggingFace import state
  const [showHFModal, setShowHFModal] = useState(false);
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<any[]>([]);
  const [hfLoading, setHfLoading] = useState(false);
  const [hfError, setHfError] = useState<string | null>(null);
  const [hfImporting, setHfImporting] = useState<string | null>(null);
  const [hfColumns, setHfColumns] = useState<string[]>([]);
  const [hfInputCol, setHfInputCol] = useState('');
  const [hfOutputCol, setHfOutputCol] = useState('');
  const [hfMappingDataset, setHfMappingDataset] = useState<string | null>(null);

  const stepsRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);

  // Load API keys from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('dspy-api-keys');
    if (saved) {
      try {
        setApiKeys(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load API keys:', e);
      }
    }
  }, []);

  // Save API keys to localStorage
  const saveApiKeys = () => {
    localStorage.setItem('dspy-api-keys', JSON.stringify(apiKeys));
    setShowSettings(false);
    loadModels();
  };

  // Check Ollama status
  const checkOllamaStatus = async () => {
    try {
      const status = await api.getOllamaStatus();
      setOllamaStatus(status);
      return status;
    } catch (e) {
      setOllamaStatus({ available: false, models_count: 0, base_url: 'http://localhost:11434' });
      return null;
    }
  };

  // Load available models
  const loadModels = async () => {
    const allModels: ModelOption[] = [];

    // Check Ollama first
    await checkOllamaStatus();

    for (const provider of PROVIDERS) {
      try {
        const models = await api.getModels(provider.id);
        models.forEach((model: string) => {
          allModels.push({
            value: `${provider.id}/${model}`,
            label: `${model} (${provider.label})`,
            provider: provider.id,
          });
        });
      } catch (e) {
        console.error(`Failed to load ${provider.id} models:`, e);
      }
    }

    setTargetModels(allModels);
    // Allow Ollama models for optimizer too (local optimization)
    setOptimizerModels(allModels.filter(m => 
      m.provider === 'openai' || m.provider === 'anthropic' || m.provider === 'ollama'
    ));

    if (allModels.length > 0 && !targetLM) {
      setTargetLM(allModels[0].value);
    }
    if (allModels.length > 0 && !optimizerLM) {
      const defaultOptimizer = allModels.find(m => m.value.includes('gpt-4o-mini'));
      setOptimizerLM(defaultOptimizer?.value || allModels[0].value);
    }
  };

  useEffect(() => {
    const init = async () => {
      await loadModels();
      // Минимальное время показа splash screen
      setTimeout(() => setIsAppLoading(false), 1500);
    };
    init();
  }, []);

  // Auto-scroll steps
  useEffect(() => {
    if (stepsRef.current) {
      stepsRef.current.scrollTop = stepsRef.current.scrollHeight;
    }
  }, [reactSteps]);

  // Parse dataset
  // Parse dataset - supports both array and {data: [...]} format
  const parsedDataset = useMemo(() => {
    try {
      const parsed = JSON.parse(datasetText);
      // If it's an object with 'data' field, extract it
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed) && Array.isArray(parsed.data)) {
        return parsed.data;
      }
      // If it's already an array, use it directly
      if (Array.isArray(parsed)) {
        return parsed;
      }
      return null;
    } catch {
      return null;
    }
  }, [datasetText]);

  const datasetCount = parsedDataset?.length || 0;

  // Run optimization
  const handleRun = async () => {
    if (!businessTask.trim() || !datasetText.trim()) return;

    // Abort previous run
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }

    setError(null);
    setIsRunning(true);
    setResult(null);
    setReactSteps([]);
    setCurrentPhase('Connecting...');

    // Validate dataset
    if (!parsedDataset || parsedDataset.length < 5) {
      setError('Invalid dataset: Need at least 5 examples as [{input, output}, ...]');
      setIsRunning(false);
      return;
    }
    const dataset = parsedDataset;

    // Initial step
    setReactSteps([{
      id: 'step_init',
      name: 'Initialize Orchestrator',
      tool: 'init',
      status: 'running',
      thought: 'Starting DSPy optimization pipeline...',
    }]);

    // Stream orchestration
    abortRef.current = api.streamOrchestrate(
      {
        business_task: businessTask,
        target_lm: targetLM,
        optimizer_lm: optimizerLM,
        dataset,
        quality_profile: qualityProfile,
        optimizer_strategy: optimizerStrategy,
        use_agent: useAgent,
      },
      {
        onStep: (step) => {
          if (!step || !step.id) return; // Skip invalid steps
          setCurrentPhase(step?.name || 'Processing...');
          setReactSteps(prev => {
            const idx = prev.findIndex(s => s.id === step.id);
            if (idx >= 0) {
              const updated = [...prev];
              updated[idx] = step;
              return updated;
            }
            // Mark init as success on first real step
            if (prev.length === 1 && prev[0].id === 'step_init') {
              return [{ ...prev[0], status: 'success' }, step];
            }
            return [...prev, step];
          });
        },
        onComplete: (response) => {
          setReactSteps(prev => prev.map(s =>
            s.status === 'running' ? { ...s, status: 'success' } : s
          ));
          setResult(response);
          setCurrentPhase('Complete!');
          setIsRunning(false);
          abortRef.current = null;
        },
        onError: (errorMsg) => {
          setError(errorMsg);
          setReactSteps(prev => {
            const last = prev[prev.length - 1];
            if (last?.status === 'running') {
              return prev.map(s => s.id === last.id ? { ...s, status: 'error', error: errorMsg } : s);
            }
            return prev;
          });
          setIsRunning(false);
          abortRef.current = null;
        },
      }
    );
  };

  // Copy code
  const handleCopy = () => {
    if (result?.program_code) {
      navigator.clipboard.writeText(result.program_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // HuggingFace search
  const handleHFSearch = async () => {
    if (!hfQuery.trim()) return;
    setHfLoading(true);
    setHfError(null);
    setHfResults([]);
    setHfMappingDataset(null);
    try {
      const res = await api.searchHFDatasets(hfQuery, 20);
      setHfResults(res.results || []);
    } catch (e) {
      setHfError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setHfLoading(false);
    }
  };

  // Start HuggingFace import - inspect columns first
  const handleHFStartImport = async (datasetId: string) => {
    setHfImporting(datasetId);
    setHfMappingDataset(datasetId);
    setHfColumns([]);
    setHfInputCol('');
    setHfOutputCol('');
    setHfError(null);
    try {
      const info = await api.inspectHFDataset({ dataset_id: datasetId });
      setHfColumns(info.columns);
      if (info.suggested_input) setHfInputCol(info.suggested_input);
      if (info.suggested_output) setHfOutputCol(info.suggested_output);
    } catch (e) {
      setHfError(e instanceof Error ? e.message : 'Failed to inspect dataset');
      setHfMappingDataset(null);
    } finally {
      setHfImporting(null);
    }
  };

  // Complete HuggingFace import
  const handleHFImport = async () => {
    if (!hfMappingDataset || !hfInputCol || !hfOutputCol) return;
    setHfImporting(hfMappingDataset);
    setHfError(null);
    try {
      const imported = await api.importHFDataset({
        dataset_id: hfMappingDataset,
        input_key: hfInputCol,
        output_key: hfOutputCol,
        max_items: 200,
      });
      setDatasetText(JSON.stringify(imported.items, null, 2));
      setShowHFModal(false);
      setHfMappingDataset(null);
      setHfResults([]);
      setHfQuery('');
    } catch (e) {
      setHfError(e instanceof Error ? e.message : 'Import failed');
    } finally {
      setHfImporting(null);
    }
  };

  // Download code
  const handleDownload = () => {
    if (result?.program_code) {
      const blob = new Blob([result.program_code], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dspy_program_${result.artifact_version_id}.py`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  // Test artifact
  const handleTest = async () => {
    if (!testInput.trim() || !result) return;
    setIsTesting(true);
    setTestOutput(null);
    try {
      const output = await api.testArtifact({
        artifact_id: result.artifact_version_id,
        input_text: testInput,
        target_lm: targetLM,
        program_code: result.program_code,
      });
      setTestOutput(output);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Test failed';
      setTestOutput(`Error: ${message}`);
    } finally {
      setIsTesting(false);
    }
  };

  // Step icon
  const getStepIcon = (status: string) => {
    switch (status) {
      case 'running':
        return (
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        );
      case 'success':
        return <Check className="w-4 h-4 text-emerald-400" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <div className="w-4 h-4 rounded-full border-2 border-white/20" />;
    }
  };

  // Splash Screen
  if (isAppLoading) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <div className="flex flex-col items-center">
          {/* Animated Logo */}
          <div className="relative mb-8">
            {/* Outer glow */}
            <div className="absolute inset-0 w-24 h-24 rounded-full bg-blue-500 opacity-30 blur-2xl animate-pulse" />
            
            {/* Orbiting particles */}
            <div className="absolute inset-0 w-24 h-24 flex items-center justify-center">
              <div className="absolute w-3 h-3 rounded-full bg-blue-400 shadow-lg shadow-blue-400/50 animate-orbit" />
              <div className="absolute w-2.5 h-2.5 rounded-full bg-blue-300 shadow-lg shadow-blue-300/50 animate-orbit-reverse" />
              <div className="absolute w-2 h-2 rounded-full bg-blue-500 shadow-lg shadow-blue-500/50 animate-orbit" style={{ animationDelay: '-2s', animationDuration: '5s' }} />
            </div>
            
            {/* Main icon */}
            <div className="relative w-24 h-24 rounded-2xl flex items-center justify-center animate-float shadow-2xl">
              <DSPyLogo className="w-16 h-16" />
            </div>
            
            {/* Spinning ring */}
            <div className="absolute -inset-4 border-2 border-dashed border-white/10 rounded-3xl animate-spin-slow" />
          </div>
          
          {/* Title */}
          <h1 className="text-3xl font-bold text-white mb-3">
            DSPy Prompt Optimizer
          </h1>
          
          {/* Subtitle */}
          <p className="text-white/40 text-sm mb-6">Automated prompt engineering</p>
          
          {/* Loading bar */}
          <div className="w-48 h-1 bg-white/10 rounded-full overflow-hidden">
            <div className="h-full bg-blue-500 rounded-full animate-shimmer" style={{ width: '100%' }} />
          </div>
          
          {/* Loading dots */}
          <div className="flex gap-1.5 mt-6">
            <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="w-2 h-2 rounded-full bg-blue-600 animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-900 text-white">
      {/* Header */}
      <header className="border-b border-white/10 bg-dark-800/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center">
              <DSPyLogo className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">DSPy Prompt Optimizer</h1>
              <p className="text-xs text-white/40">Automated prompt engineering</p>
            </div>
          </div>
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
          >
            <Settings className="w-5 h-5 text-white/60" />
          </button>
        </div>
      </header>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-dark-800 border border-white/10 rounded-2xl w-full max-w-md p-6">
            <h2 className="text-lg font-semibold mb-4">API Keys</h2>
            <div className="space-y-4">
              {PROVIDERS.filter(p => p.requiresKey).map(provider => (
                <div key={provider.id}>
                  <label className="text-xs text-white/50 uppercase tracking-wider">{provider.label}</label>
                  <input
                    type="password"
                    value={apiKeys[provider.id] || ''}
                    onChange={(e) => setApiKeys(prev => ({ ...prev, [provider.id]: e.target.value }))}
                    placeholder={`Enter ${provider.label} API key`}
                    className="w-full mt-1 bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-white/20"
                  />
                </div>
              ))}
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowSettings(false)}
                className="flex-1 px-4 py-2 rounded-lg border border-white/10 text-white/60 hover:bg-white/5"
              >
                Cancel
              </button>
              <button
                onClick={saveApiKeys}
                className="flex-1 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column: Configuration */}
          <div className="col-span-3 space-y-4">
            {/* Target Model */}
            <div className="bg-dark-800 border border-white/10 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-blue-400" />
                <span className="text-xs font-medium text-white/60 uppercase tracking-wider">Target Model</span>
              </div>
              <select
                value={targetLM}
                onChange={(e) => setTargetLM(e.target.value)}
                className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-white/20 appearance-none cursor-pointer"
              >
                {targetModels.length === 0 ? (
                  <option value="">Configure API keys first</option>
                ) : (
                  targetModels.map(m => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))
                )}
              </select>
              <p className="text-[10px] text-white/30 mt-2">Model for production inference</p>
            </div>

            {/* Optimizer Model */}
            <div className="bg-dark-800 border border-white/10 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-4 h-4 text-purple-400" />
                <span className="text-xs font-medium text-white/60 uppercase tracking-wider">Optimizer Model</span>
              </div>
              <select
                value={optimizerLM}
                onChange={(e) => setOptimizerLM(e.target.value)}
                className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-white/20 appearance-none cursor-pointer"
              >
                {optimizerModels.length === 0 ? (
                  <option value="">Configure API keys first</option>
                ) : (
                  optimizerModels.map(m => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))
                )}
              </select>
              <p className="text-[10px] text-white/30 mt-2">Model for prompt optimization</p>
            </div>

            {/* Ollama Status */}
            <div className={`border rounded-xl p-4 ${
              ollamaStatus?.available 
                ? 'bg-emerald-500/5 border-emerald-500/20' 
                : 'bg-dark-800 border-white/10'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-white/60 uppercase tracking-wider">Ollama (Local)</span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full ${
                  ollamaStatus?.available 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : 'bg-red-500/20 text-red-400'
                }`}>
                  {ollamaStatus?.available ? 'Connected' : 'Offline'}
                </span>
              </div>
              {ollamaStatus?.available ? (
                <p className="text-[11px] text-white/50">
                  {ollamaStatus.models_count} model{ollamaStatus.models_count !== 1 ? 's' : ''} available
                </p>
              ) : (
                <p className="text-[10px] text-white/40">
                  Start Ollama to use local models
                </p>
              )}
              <button
                onClick={() => loadModels()}
                className="mt-2 text-[10px] text-blue-400 hover:text-blue-300"
              >
                ↻ Refresh models
              </button>
            </div>

            {/* Quality Profile */}
            <div className="bg-dark-800 border border-white/10 rounded-xl p-4">
              <span className="text-xs font-medium text-white/60 uppercase tracking-wider">Quality Profile</span>
              <div className="grid grid-cols-3 gap-2 mt-3">
                {QUALITY_PROFILES.map(p => (
                  <button
                    key={p.value}
                    onClick={() => setQualityProfile(p.value)}
                    className={`px-2 py-1.5 text-xs rounded-lg border transition-all ${
                      qualityProfile === p.value
                        ? 'bg-white/10 border-white/20 text-white'
                        : 'border-white/5 text-white/50 hover:bg-white/5'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-white/30 mt-2">
                {QUALITY_PROFILES.find(p => p.value === qualityProfile)?.description}
              </p>
            </div>

            {/* LangChain Agent Toggle */}
            <div className="bg-dark-800 border border-white/10 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-medium text-white/60 uppercase tracking-wider">LangChain Agent</span>
                  <p className="text-[10px] text-white/30 mt-1">
                    {useAgent ? 'Agent analyzes task and configures DSPy' : 'Direct DSPy optimization'}
                  </p>
                </div>
                <button
                  onClick={() => setUseAgent(!useAgent)}
                  className={`relative w-11 h-6 rounded-full transition-colors ${
                    useAgent ? 'bg-emerald-500' : 'bg-white/20'
                  }`}
                >
                  <span className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                    useAgent ? 'left-6' : 'left-1'
                  }`} />
                </button>
              </div>
            </div>

            {/* Optimizer Strategy */}
            <div className="bg-dark-800 border border-white/10 rounded-xl p-4">
              <span className="text-xs font-medium text-white/60 uppercase tracking-wider">Optimizer</span>
              <div className="grid grid-cols-2 gap-2 mt-3">
                {OPTIMIZER_STRATEGIES.map(s => (
                  <button
                    key={s.value}
                    onClick={() => setOptimizerStrategy(s.value)}
                    className={`px-2 py-1.5 text-xs rounded-lg border transition-all ${
                      optimizerStrategy === s.value
                        ? 'bg-white/10 border-white/20 text-white'
                        : 'border-white/5 text-white/50 hover:bg-white/5'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-white/30 mt-2">
                {OPTIMIZER_STRATEGIES.find(s => s.value === optimizerStrategy)?.description}
              </p>
            </div>

            {/* Tips */}
            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-4">
              <div className="text-[10px] font-medium text-emerald-400 uppercase tracking-wider mb-2">Best for DSPy</div>
              <ul className="text-[11px] text-white/50 space-y-1">
                <li>• Complex multi-step pipelines</li>
                <li>• 30-50+ labeled examples</li>
                <li>• Clear success metrics</li>
              </ul>
            </div>
          </div>

          {/* Middle Column: Input + Steps */}
          <div className="col-span-5 space-y-4">
            {/* Task Input */}
            <div className="bg-dark-800 border border-white/10 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-white/5 bg-white/[0.02]">
                <h2 className="text-sm font-medium text-white/80">Business Task</h2>
                <p className="text-[10px] text-white/40 mt-0.5">Describe what you want the model to do</p>
              </div>
              <div className="p-4">
                <textarea
                  value={businessTask}
                  onChange={(e) => setBusinessTask(e.target.value)}
                  placeholder="Example: Analyze customer support tickets and classify them by urgency (low, medium, high) with a brief explanation..."
                  className="w-full h-28 bg-black/30 border border-white/5 rounded-xl px-4 py-3 text-sm text-white/90 placeholder:text-white/30 resize-none focus:outline-none focus:border-white/15"
                />
              </div>
            </div>

            {/* Dataset Input */}
            <div className="bg-dark-800 border border-white/10 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-white/5 bg-white/[0.02] flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4 text-white/40" />
                    <h2 className="text-sm font-medium text-white/80">Training Dataset</h2>
                  </div>
                  <p className="text-[10px] text-white/40 mt-0.5">JSON array of {`{input, output}`} examples</p>
                </div>
                <div className="flex items-center gap-2">
                  {datasetCount > 0 && (
                    <span className="text-xs px-2 py-1 rounded-full bg-white/5 text-white/50">
                      {datasetCount} examples
                    </span>
                  )}
                  <button
                    onClick={() => setShowHFModal(true)}
                    className="text-xs px-3 py-1.5 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 hover:bg-yellow-500/20 transition-colors flex items-center gap-1.5"
                  >
                    <span className="text-base">🤗</span>
                    Import from HF
                  </button>
                </div>
              </div>
              <div className="p-4">
                <textarea
                  value={datasetText}
                  onChange={(e) => setDatasetText(e.target.value)}
                  placeholder={`[\n  {"input": "Customer complaint about late delivery", "output": "high"},\n  {"input": "Question about product features", "output": "low"},\n  ...\n]`}
                  className="w-full h-36 bg-black/30 border border-white/5 rounded-xl px-4 py-3 text-sm font-mono text-white/90 placeholder:text-white/30 resize-none focus:outline-none focus:border-white/15"
                />
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRun}
              disabled={isRunning || !businessTask.trim() || datasetCount < 5}
              className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all ${
                isRunning
                  ? 'bg-blue-600/50 cursor-wait'
                  : !businessTask.trim() || datasetCount < 5
                  ? 'bg-white/5 text-white/30 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white shadow-lg shadow-blue-500/20'
              }`}
            >
              {isRunning ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {currentPhase}
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run DSPy Optimization
                </>
              )}
            </button>

            {/* ReAct Steps */}
            <div className="bg-dark-800 border border-white/10 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-white/5 bg-white/[0.02] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FlaskConical className="w-4 h-4 text-white/40" />
                  <h2 className="text-sm font-medium text-white/80">ReAct Steps</h2>
                </div>
                <span className="text-[10px] text-white/30">{reactSteps.length} steps</span>
              </div>
              
              {error && (
                <div className="mx-4 mt-3 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-300 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}
              
              <div ref={stepsRef} className="max-h-64 overflow-y-auto p-4 space-y-2 custom-scrollbar">
                {reactSteps.length === 0 && !isRunning ? (
                  <div className="text-center py-8 text-white/30 text-sm">
                    <Sparkles className="w-8 h-8 mx-auto mb-2 text-white/10" />
                    <p>Agent steps will appear here</p>
                  </div>
                ) : reactSteps.length === 0 && isRunning ? (
                  /* Beautiful Loading Animation */
                  <div className="py-8 flex flex-col items-center justify-center">
                    {/* Animated Logo */}
                    <div className="relative mb-6">
                      {/* Outer glow ring */}
                      <div className="absolute inset-0 w-20 h-20 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 opacity-20 blur-xl animate-pulse" />
                      
                      {/* Orbiting particles */}
                      <div className="absolute inset-0 w-20 h-20 flex items-center justify-center">
                        <div className="absolute w-3 h-3 rounded-full bg-blue-400 animate-orbit" />
                        <div className="absolute w-2 h-2 rounded-full bg-purple-400 animate-orbit-reverse" />
                        <div className="absolute w-1.5 h-1.5 rounded-full bg-cyan-400 animate-orbit" style={{ animationDelay: '-2s' }} />
                      </div>
                      
                      {/* Main icon container */}
                      <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center animate-pulse-glow animate-float">
                        <DSPyLogo className="w-14 h-14" />
                      </div>
                      
                      {/* Spinning ring */}
                      <div className="absolute -inset-2 border-2 border-dashed border-white/10 rounded-3xl animate-spin-slow" />
                    </div>
                    
                    {/* Animated Title */}
                    <h3 className="text-lg font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent animate-text-gradient mb-2">
                      DSPy Prompt Optimizer
                    </h3>
                    
                    {/* Status text with shimmer */}
                    <div className="relative overflow-hidden px-4 py-1.5 rounded-full bg-white/5 border border-white/10">
                      <span className="text-xs text-white/60">{currentPhase || 'Initializing...'}</span>
                      <div className="absolute inset-0 animate-shimmer" />
                    </div>
                    
                    {/* Animated dots */}
                    <div className="flex gap-1 mt-4">
                      <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    
                    {/* Subtle hint */}
                    <p className="text-[10px] text-white/30 mt-4">
                      Preparing optimization pipeline...
                    </p>
                  </div>
                ) : (
                  reactSteps.map(step => (
                    <div
                      key={step.id}
                      className={`border rounded-lg p-3 transition-all animate-fadeIn ${
                        step.status === 'running' ? 'border-blue-500/30 bg-blue-500/5' :
                        step.status === 'success' ? 'border-white/10 bg-white/[0.02]' :
                        step.status === 'error' ? 'border-red-500/30 bg-red-500/5' :
                        'border-white/5'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <div className="mt-0.5">{getStepIcon(step.status)}</div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-medium text-white/80">{step.name}</span>
                            <span className="text-[10px] font-mono text-white/30">{step.tool}</span>
                          </div>
                          {step.thought && (
                            <p className="text-[11px] text-white/50 mt-1 italic">💭 {step.thought}</p>
                          )}
                          {step.action && (
                            <p className="text-[10px] font-mono text-blue-400/70 mt-1 bg-black/30 px-2 py-1 rounded">
                              → {step.action}
                            </p>
                          )}
                          {step.observation && (
                            <p className="text-[10px] font-mono text-emerald-400/70 mt-1">
                              ← {step.observation}
                            </p>
                          )}
                          {step.error && (
                            <p className="text-[10px] text-red-400 mt-1">✗ {step.error}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="col-span-4">
            <div className="bg-dark-800 border border-white/10 rounded-xl overflow-hidden sticky top-24">
              <div className="px-4 py-3 border-b border-white/5 bg-white/[0.02]">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-medium text-white/80">Results</h2>
                  {result?.eval_results?.metric_value !== undefined && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      {(result.eval_results.metric_value * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                {result && (
                  <div className="flex gap-1 mt-3">
                    {(['overview', 'code', 'test', 'deploy'] as const).map(tab => (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-3 py-1 text-[11px] rounded-md transition-all ${
                          activeTab === tab
                            ? 'bg-white/10 text-white'
                            : 'text-white/40 hover:text-white/60'
                        }`}
                      >
                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="p-4 max-h-[calc(100vh-200px)] overflow-y-auto custom-scrollbar">
                {!result ? (
                  <div className="text-center py-12 text-white/30">
                    <Rocket className="w-12 h-12 mx-auto mb-3 text-white/10" />
                    <p className="text-sm">No results yet</p>
                    <p className="text-xs text-white/20 mt-1">Run optimization to see results</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {activeTab === 'overview' && (
                      <>
                        {/* Metrics */}
                        <div className="grid grid-cols-2 gap-2">
                          <div className="bg-white/5 rounded-xl p-3 border border-white/5">
                            <div className="text-[10px] text-white/40 uppercase">Metric</div>
                            <div className="text-xl font-bold text-white/90 mt-1">
                              {((result.eval_results?.metric_value ?? result.eval_results?.test_accuracy ?? result.eval_results?.dev_accuracy ?? 0) * 100).toFixed(1)}%
                            </div>
                            <div className="text-[10px] text-white/30">{result.eval_results?.metric_name || 'accuracy'}</div>
                          </div>
                          <div className="bg-white/5 rounded-xl p-3 border border-white/5">
                            <div className="text-[10px] text-white/40 uppercase">Steps</div>
                            <div className="text-xl font-bold text-white/90 mt-1">{result.react_iterations}</div>
                            <div className="text-[10px] text-white/30">ReAct iterations</div>
                          </div>
                        </div>

                        {/* Details */}
                        <div className="bg-white/5 rounded-xl p-3 border border-white/5 space-y-2">
                          <div className="text-[10px] text-white/40 uppercase mb-2">Details</div>
                          <div className="flex justify-between text-[11px]">
                            <span className="text-white/50">Task Type</span>
                            <span className="text-white/80">{result.task_analysis?.task_type || 'N/A'}</span>
                          </div>
                          <div className="flex justify-between text-[11px]">
                            <span className="text-white/50">Domain</span>
                            <span className="text-white/80">{result.task_analysis?.domain || 'N/A'}</span>
                          </div>
                          <div className="flex justify-between text-[11px]">
                            <span className="text-white/50">Optimizer</span>
                            <span className="text-white/80">{result.optimizer_type}</span>
                          </div>
                          <div className="flex justify-between text-[11px]">
                            <span className="text-white/50">Real DSPy</span>
                            <span className="text-white/80">{result.eval_results?.real_dspy ? 'Yes' : 'No'}</span>
                          </div>
                        </div>

                        {/* Artifact */}
                        <div className="bg-white/5 rounded-xl p-3 border border-white/5">
                          <div className="text-[10px] text-white/40 uppercase mb-2">Artifact ID</div>
                          <code className="text-xs text-white/60 bg-black/30 px-2 py-1 rounded block">
                            {result.artifact_version_id}
                          </code>
                        </div>
                      </>
                    )}

                    {activeTab === 'code' && (
                      <div className="space-y-3">
                        <div className="flex gap-2">
                          <button
                            onClick={handleCopy}
                            className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs transition-all"
                          >
                            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                            {copied ? 'Copied!' : 'Copy'}
                          </button>
                          <button
                            onClick={handleDownload}
                            className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs transition-all"
                          >
                            <Download className="w-3 h-3" />
                            Download
                          </button>
                        </div>
                        <pre className="text-[11px] font-mono text-white/70 bg-black/40 p-3 rounded-xl border border-white/5 overflow-x-auto whitespace-pre-wrap">
                          {result.program_code}
                        </pre>
                      </div>
                    )}

                    {activeTab === 'test' && (
                      <div className="space-y-3">
                        <div>
                          <label className="text-[11px] text-white/50">Test Input</label>
                          <textarea
                            value={testInput}
                            onChange={(e) => setTestInput(e.target.value)}
                            placeholder="e.g. I want to cancel my subscription"
                            className="w-full h-20 mt-1 bg-black/30 border border-white/10 rounded-xl px-3 py-2 text-sm resize-none focus:outline-none focus:border-white/20"
                          />
                        </div>
                        <button
                          onClick={handleTest}
                          disabled={isTesting || !testInput.trim()}
                          className="w-full py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-white/5 disabled:text-white/30 text-sm font-medium transition-all"
                        >
                          {isTesting ? 'Running...' : 'Run Test'}
                        </button>
                        {testOutput && (
                          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3">
                            <div className="text-[10px] text-emerald-400 uppercase mb-1">Output</div>
                            <pre className="text-sm text-emerald-300 whitespace-pre-wrap">{testOutput}</pre>
                          </div>
                        )}
                      </div>
                    )}

                    {activeTab === 'deploy' && (
                      <div className="space-y-2">
                        <button
                          onClick={handleDownload}
                          className="w-full py-2 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-sm font-medium flex items-center justify-center gap-2"
                        >
                          <Download className="w-3 h-3" />
                          Download
                        </button>
                        
                        <div className="bg-white/5 rounded-lg p-2 border border-white/5">
                          <div className="text-[9px] text-white/40 uppercase mb-1">Artifact</div>
                          <a 
                            href={`file:///Users/artemk/Desktop/dspy/backend/data/artifacts/${result.artifact_version_id}`}
                            className="text-[10px] font-mono text-white/60 hover:text-white/80 block truncate"
                            title={`/Users/artemk/Desktop/dspy/backend/data/artifacts/${result.artifact_version_id}`}
                          >
                            {result.artifact_version_id}
                          </a>
                        </div>

                        <div className="bg-white/5 rounded-lg p-2 border border-white/5 max-h-[200px] overflow-y-auto">
                          <div className="text-[9px] text-white/40 uppercase mb-1">Usage</div>
                          <pre className="text-[9px] font-mono text-white/60 whitespace-pre-wrap">{`import dspy
lm = dspy.LM("ollama_chat/model")
dspy.configure(lm=lm)

from program import OptimizedModule
module = OptimizedModule()
result = module(input="text")
print(result)`}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* HuggingFace Import Modal */}
      {showHFModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-2xl max-h-[80vh] bg-[#0a0a0f] border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
            {/* Header */}
            <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between bg-white/[0.02]">
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-white/30 font-semibold mb-1">
                  Dataset Catalog
                </div>
                <div className="text-sm font-semibold text-white/90 flex items-center gap-2">
                  <span className="text-lg">🤗</span>
                  Import from Hugging Face
                </div>
              </div>
              <button
                onClick={() => { setShowHFModal(false); setHfMappingDataset(null); setHfResults([]); setHfQuery(''); }}
                className="p-1.5 rounded-lg border border-white/10 text-white/60 hover:text-white hover:border-white/30 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Search */}
            <div className="p-4 border-b border-white/10 flex items-center gap-3">
              <input
                type="text"
                value={hfQuery}
                onChange={(e) => setHfQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleHFSearch(); }}
                placeholder="Search datasets (e.g., customer support, sentiment, qa)..."
                className="flex-1 bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/90 placeholder:text-white/30 focus:outline-none focus:border-white/20"
              />
              <button
                onClick={handleHFSearch}
                disabled={hfLoading || !hfQuery.trim()}
                className="px-4 py-2 text-sm rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {hfLoading ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                Search
              </button>
            </div>

            {/* Results */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {hfLoading && (
                <div className="text-xs text-white/40 text-center py-4">Searching Hugging Face...</div>
              )}
              {hfError && (
                <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">{hfError}</div>
              )}
              {!hfLoading && !hfError && hfResults.length === 0 && (
                <div className="text-xs text-white/40 text-center py-8">
                  Enter a query to search public datasets on Hugging Face Hub
                </div>
              )}
              {hfResults.map((ds) => (
                <div
                  key={ds.id}
                  className="bg-white/[0.02] border border-white/10 rounded-lg px-4 py-3 flex items-start justify-between gap-3 hover:bg-white/[0.04] transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">{ds.id}</div>
                    {ds.cardData?.pretty_name && (
                      <div className="text-[11px] text-white/60 truncate mt-0.5">{ds.cardData.pretty_name}</div>
                    )}
                    <div className="text-[10px] text-white/40 mt-1 flex gap-3">
                      {typeof ds.downloads === 'number' && <span>{ds.downloads.toLocaleString()} downloads</span>}
                      {typeof ds.likes === 'number' && <span>{ds.likes} likes</span>}
                    </div>
                  </div>
                  <button
                    onClick={() => handleHFStartImport(ds.id)}
                    disabled={hfImporting === ds.id}
                    className="px-3 py-1.5 text-xs rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 hover:bg-yellow-500/20 disabled:opacity-50 shrink-0"
                  >
                    {hfImporting === ds.id ? 'Loading...' : 'Select'}
                  </button>
                </div>
              ))}
            </div>

            {/* Column Mapping */}
            {hfMappingDataset && hfColumns.length > 0 && (
              <div className="px-5 py-4 border-t border-white/10 bg-white/[0.02] space-y-3">
                <div className="text-xs text-white/60">
                  Map columns for <span className="font-semibold text-white/80">{hfMappingDataset}</span>
                </div>
                <div className="flex gap-3">
                  <div className="flex-1">
                    <label className="text-[10px] text-white/40 uppercase mb-1 block">Input Column</label>
                    <select
                      value={hfInputCol}
                      onChange={(e) => setHfInputCol(e.target.value)}
                      className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/90 focus:outline-none"
                    >
                      <option value="">Select...</option>
                      {hfColumns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex-1">
                    <label className="text-[10px] text-white/40 uppercase mb-1 block">Output Column</label>
                    <select
                      value={hfOutputCol}
                      onChange={(e) => setHfOutputCol(e.target.value)}
                      className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/90 focus:outline-none"
                    >
                      <option value="">Select...</option>
                      {hfColumns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <button
                  onClick={handleHFImport}
                  disabled={!hfInputCol || !hfOutputCol || !!hfImporting}
                  className="w-full py-2.5 text-sm rounded-lg bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-500 hover:to-orange-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {hfImporting ? 'Importing...' : 'Import Dataset'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
