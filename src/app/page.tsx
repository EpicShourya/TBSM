"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  AlertTriangle, 
  CheckCircle, 
  Brain, 
  Target, 
  TrendingUp, 
  FileText,
  Clock,
  Activity,
  Loader2
} from 'lucide-react';

interface DetectionResult {
  hallucination_probability: number;
  confidence_score: number;
  detected_issues: string[];
  recommendations: string[];
  is_safe: boolean;
  detailed_metrics: {
    confidence_issues: number;
    factual_density: number;
    contradiction_score: number;
    ml_probability: number;
    vagueness: number;
    absoluteness: number;
    credibility_markers: number;
    speculation: number;
    complexity: number;
  };
  hallucination_ratio: number;
  risk_level: string;
  text_quality_score: number;
  credibility_indicators: Record<string, boolean | number>;
  linguistic_analysis: Record<string, number>;
  content_analysis: Record<string, number>;
  accuracy_percentage: number;
  prediction_confidence: number;
  reliability_score: number;
  timestamp?: string;
  processing_time?: number;
}

export default function Home() {
  const [text, setText] = useState('');
  const [context, setContext] = useState('');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeText = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text: text.trim(), 
          context: context.trim() || undefined 
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setError(error instanceof Error ? error.message : 'Failed to analyze text');
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevel = (probability: number) => {
    if (probability < 0.3) return { label: 'Low Risk', color: 'text-green-400', bg: 'bg-green-500/10' };
    if (probability < 0.6) return { label: 'Medium Risk', color: 'text-yellow-400', bg: 'bg-yellow-500/10' };
    return { label: 'High Risk', color: 'text-red-400', bg: 'bg-red-500/10' };
  };

  const getRiskLevelFromString = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low':
        return { label: 'Low Risk', color: 'text-green-400', bg: 'bg-green-500/10', border: 'border-green-500/30' };
      case 'medium':
        return { label: 'Medium Risk', color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/30' };
      case 'high':
        return { label: 'High Risk', color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30' };
      case 'critical':
        return { label: 'Critical Risk', color: 'text-red-300', bg: 'bg-red-500/20', border: 'border-red-400/50' };
      default:
        return { label: 'Unknown Risk', color: 'text-gray-400', bg: 'bg-gray-500/10', border: 'border-gray-500/30' };
    }
  };

  const exampleTexts = [
    {
      text: "The Eiffel Tower is located in Paris, France and was built in 1889.",
      label: "Factual Example"
    },
    {
      text: "I am absolutely certain that the moon is made of cheese and everyone knows this fact.",
      label: "Hallucination Example"
    },
    {
      text: "According to research, water boils at 100 degrees Celsius at sea level.",
      label: "Scientific Fact"
    }
  ];

  // Prepare data for charts
  const radarData = result ? [
    { metric: 'Accuracy', value: result.accuracy_percentage },
    { metric: 'Reliability', value: result.reliability_score },
    { metric: 'Factual Density', value: result.detailed_metrics.factual_density * 100 },
    { metric: 'Credibility', value: result.detailed_metrics.credibility_markers * 100 },
    { metric: 'Logic Consistency', value: (1 - result.detailed_metrics.contradiction_score) * 100 },
    { metric: 'Language Clarity', value: (1 - result.detailed_metrics.vagueness) * 100 },
  ] : [];

  const trendData = result ? [
    { name: 'Overall Risk', risk: result.hallucination_probability * 100 },
    { name: 'ML Model', risk: result.detailed_metrics.ml_probability * 100 },
    { name: 'Language Issues', risk: result.detailed_metrics.confidence_issues * 100 },
    { name: 'Vagueness', risk: result.detailed_metrics.vagueness * 100 },
    { name: 'Speculation', risk: result.detailed_metrics.speculation * 100 },
    { name: 'Contradictions', risk: result.detailed_metrics.contradiction_score * 100 },
  ] : [];

  const pieData = result ? [
    { name: 'Safe', value: result.confidence_score * 100, color: '#10b981' },
    { name: 'Risk', value: result.hallucination_probability * 100, color: '#ef4444' },
  ] : [];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center space-x-4">
            <Brain className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                Hallucination Detection Dashboard
              </h1>
              <p className="text-muted-foreground">
                Advanced AI text analysis and credibility assessment
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Section */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="border-border">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="h-5 w-5" />
                  <span>Text Analysis</span>
                </CardTitle>
                <CardDescription>
                  Enter text to analyze for potential hallucinations
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="text">Text to Analyze</Label>
                  <Textarea
                    id="text"
                    value={text}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setText(e.target.value)}
                    placeholder="Enter the text you want to analyze..."
                    className="mt-2 min-h-32 resize-none bg-background"
                    maxLength={10000}
                  />
                  <div className="text-xs text-muted-foreground mt-2">
                    {text.length}/10,000 characters
                  </div>
                </div>

                <div>
                  <Label htmlFor="context">Context (Optional)</Label>
                  <Textarea
                    id="context"
                    value={context}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setContext(e.target.value)}
                    placeholder="Provide additional context..."
                    className="mt-2 min-h-20 resize-none bg-background"
                    maxLength={5000}
                  />
                </div>

                <Button
                  onClick={analyzeText}
                  disabled={loading || !text.trim()}
                  className="w-full"
                  size="lg"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Target className="mr-2 h-4 w-4" />
                      Analyze Text
                    </>
                  )}
                </Button>

                {error && (
                  <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="h-4 w-4 text-destructive" />
                      <span className="font-medium text-destructive">Error</span>
                    </div>
                    <p className="text-sm text-destructive/80 mt-1">{error}</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Examples */}
            <Card className="border-border">
              <CardHeader>
                <CardTitle className="text-lg">Quick Examples</CardTitle>
                <CardDescription>
                  Click to load example texts
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {exampleTexts.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setText(example.text)}
                    className="w-full text-left p-3 border border-border rounded-lg hover:bg-accent transition-colors"
                  >
                    <div className="font-medium text-sm text-primary mb-1">
                      {example.label}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {example.text.substring(0, 80)}...
                    </div>
                  </button>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2 space-y-6">
            {result ? (
              <>
                {/* Main Results */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  {/* Risk Level Card - Enhanced with proper colors */}
                  <Card className={`border-2 ${result.risk_level ? getRiskLevelFromString(result.risk_level).bg : getRiskLevel(result.hallucination_probability).bg} ${result.risk_level ? getRiskLevelFromString(result.risk_level).border : 'border-border'}`}>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Risk Level</p>
                          <p className={`text-3xl font-bold ${result.risk_level ? getRiskLevelFromString(result.risk_level).color : getRiskLevel(result.hallucination_probability).color}`}>
                            {(result.hallucination_probability * 100).toFixed(1)}%
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            Hallucination Probability
                          </p>
                        </div>
                        {result.is_safe ? (
                          <CheckCircle className="h-8 w-8 text-green-400" />
                        ) : (
                          <AlertTriangle className={`h-8 w-8 ${result.risk_level ? getRiskLevelFromString(result.risk_level).color : 'text-red-400'}`} />
                        )}
                      </div>
                      <div className="mt-4">
                        <Badge 
                          variant={result.is_safe ? "secondary" : "destructive"}
                          className={`text-xs font-semibold ${result.risk_level ? getRiskLevelFromString(result.risk_level).color : getRiskLevel(result.hallucination_probability).color}`}
                        >
                          {result.risk_level ? getRiskLevelFromString(result.risk_level).label : getRiskLevel(result.hallucination_probability).label}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border-border">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Accuracy</p>
                          <p className="text-2xl font-bold text-green-400">
                            {result.accuracy_percentage.toFixed(1)}%
                          </p>
                        </div>
                        <Target className="h-8 w-8 text-green-400" />
                      </div>
                      <div className="mt-4">
                        <Progress 
                          value={result.accuracy_percentage} 
                          className="h-2"
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border-border">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Confidence</p>
                          <p className="text-2xl font-bold text-blue-400">
                            {(result.confidence_score * 100).toFixed(1)}%
                          </p>
                        </div>
                        <Activity className="h-8 w-8 text-blue-400" />
                      </div>
                      <div className="mt-4">
                        <Progress 
                          value={result.confidence_score * 100} 
                          className="h-2"
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border-border">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Reliability</p>
                          <p className="text-2xl font-bold text-purple-400">
                            {result.reliability_score.toFixed(1)}%
                          </p>
                        </div>
                        <Brain className="h-8 w-8 text-purple-400" />
                      </div>
                      <div className="mt-4">
                        <Badge variant="outline" className="text-xs">
                          {result.risk_level} Risk
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Risk Level Summary */}
                {result.risk_level && (
                  <Card className={`border-2 ${getRiskLevelFromString(result.risk_level).border} ${getRiskLevelFromString(result.risk_level).bg}`}>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-foreground mb-2">Overall Risk Assessment</h3>
                          <div className={`text-4xl font-bold ${getRiskLevelFromString(result.risk_level).color} mb-2`}>
                            {result.risk_level.toUpperCase()} RISK
                          </div>
                          <p className="text-sm text-muted-foreground">
                            Hallucination Ratio: <span className="font-semibold">{(result.hallucination_ratio * 100).toFixed(1)}%</span>
                          </p>
                          <p className="text-sm text-muted-foreground">
                            Reliability Score: <span className="font-semibold">{result.reliability_score.toFixed(1)}%</span>
                          </p>
                        </div>
                        <div className="text-right">
                          <AlertTriangle className={`h-12 w-12 ${getRiskLevelFromString(result.risk_level).color} mb-2`} />
                          <div className="text-sm text-muted-foreground">
                            Quality Score: <span className="font-semibold">{result.text_quality_score.toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Charts Section */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Risk Trend Chart */}
                  <Card className="border-border">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <TrendingUp className="h-5 w-5" />
                        <span>Risk Analysis</span>
                      </CardTitle>
                      <CardDescription>
                        Risk levels across different metrics
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={trendData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis 
                            dataKey="name" 
                            stroke="#9CA3AF"
                            fontSize={12}
                          />
                          <YAxis 
                            stroke="#9CA3AF"
                            fontSize={12}
                            domain={[0, 100]}
                          />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: '#1F2937', 
                              border: '1px solid #ef4444',
                              borderRadius: '8px',
                              color: '#ffffff'
                            }}
                            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Risk Level']}
                            labelStyle={{ color: '#ef4444' }}
                          />
                          <Area 
                            type="monotone" 
                            dataKey="risk" 
                            stroke="#dc2626" 
                            fill="#ef4444"
                            fillOpacity={0.4}
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Radar Chart */}
                  <Card className="border-border">
                    <CardHeader>
                      <CardTitle>Metric Breakdown</CardTitle>
                      <CardDescription>
                        Detailed analysis across all dimensions
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={250}>
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="#374151" />
                          <PolarAngleAxis 
                            dataKey="metric" 
                            tick={{ fill: '#9CA3AF', fontSize: 12 }}
                          />
                          <PolarRadiusAxis 
                            tick={{ fill: '#9CA3AF', fontSize: 10 }}
                            domain={[0, 100]}
                          />
                          <Radar
                            name="Score"
                            dataKey="value"
                            stroke="#3b82f6"
                            fill="#3b82f6"
                            fillOpacity={0.3}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>

                {/* Detailed Metrics */}
                <Card className="border-border">
                  <CardHeader>
                    <CardTitle>Enhanced Metrics</CardTitle>
                    <CardDescription>
                      Comprehensive analysis breakdown with accuracy assessment
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { 
                          key: 'accuracy_percentage', 
                          label: 'Overall Accuracy', 
                          value: result.accuracy_percentage / 100,
                          description: 'Calculated accuracy based on multiple factors'
                        },
                        { 
                          key: 'reliability_score', 
                          label: 'Reliability Score', 
                          value: result.reliability_score / 100,
                          description: 'Overall reliability and trustworthiness'
                        },
                        { 
                          key: 'hallucination_ratio', 
                          label: 'Hallucination Ratio', 
                          value: result.hallucination_ratio,
                          description: 'Comprehensive hallucination risk assessment'
                        },
                        { 
                          key: 'prediction_confidence', 
                          label: 'Prediction Confidence', 
                          value: result.prediction_confidence / 100,
                          description: 'Confidence in the analysis results'
                        },
                        { 
                          key: 'text_quality_score', 
                          label: 'Text Quality', 
                          value: result.text_quality_score,
                          description: 'Overall quality and clarity of text'
                        },
                        { 
                          key: 'ml_probability', 
                          label: 'ML Model Score', 
                          value: result.detailed_metrics.ml_probability,
                          description: 'Machine learning model confidence'
                        },
                        { 
                          key: 'factual_density', 
                          label: 'Factual Content', 
                          value: result.detailed_metrics.factual_density,
                          description: 'Density of factual information'
                        },
                        { 
                          key: 'credibility_markers', 
                          label: 'Source Credibility', 
                          value: result.detailed_metrics.credibility_markers,
                          description: 'Presence of credible source indicators'
                        },
                        { 
                          key: 'vagueness', 
                          label: 'Language Vagueness', 
                          value: result.detailed_metrics.vagueness,
                          description: 'Use of vague or imprecise language'
                        },
                        { 
                          key: 'speculation', 
                          label: 'Speculation Level', 
                          value: result.detailed_metrics.speculation,
                          description: 'Speculative language and claims'
                        },
                      ].map((metric) => (
                        <div key={metric.key} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <div>
                              <span className="font-medium">{metric.label}</span>
                              <p className="text-xs text-muted-foreground">
                                {metric.description}
                              </p>
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {(metric.value * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <Progress value={metric.value * 100} className="h-2" />
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Issues and Recommendations */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Credibility Indicators */}
                  <Card className="border-border">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2 text-blue-400">
                        <CheckCircle className="h-5 w-5" />
                        <span>Credibility Indicators</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {Object.entries(result.credibility_indicators).map(([key, value]) => (
                          <div key={key} className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground capitalize">
                              {key.replace(/_/g, ' ')}
                            </span>
                            <div className={`w-3 h-3 rounded-full ${
                              typeof value === 'boolean' 
                                ? (value ? 'bg-green-400' : 'bg-red-400')
                                : (value > 0.5 ? 'bg-green-400' : 'bg-red-400')
                            }`} />
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {result.detected_issues.length > 0 && (
                    <Card className="border-border">
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-2 text-red-400">
                          <AlertTriangle className="h-5 w-5" />
                          <span>Detected Issues</span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-2">
                          {result.detected_issues.map((issue, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0" />
                              <span className="text-sm text-muted-foreground">{issue}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}

                  {result.recommendations.length > 0 && (
                    <Card className="border-border">
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-2 text-green-400">
                          <CheckCircle className="h-5 w-5" />
                          <span>Recommendations</span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-2">
                          {result.recommendations.map((rec, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0" />
                              <span className="text-sm text-muted-foreground">{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </>
            ) : (
              <Card className="border-border border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-16">
                  <Brain className="h-16 w-16 text-muted-foreground/50 mb-4" />
                  <h3 className="text-xl font-semibold text-muted-foreground mb-2">
                    Ready to Analyze
                  </h3>
                  <p className="text-muted-foreground text-center max-w-md">
                    Enter some text in the input panel to get started with AI-powered 
                    hallucination detection and credibility analysis.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
