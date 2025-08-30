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
  };
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
    { metric: 'ML Score', value: result.detailed_metrics.ml_probability * 100 },
    { metric: 'Confidence Issues', value: result.detailed_metrics.confidence_issues * 100 },
    { metric: 'Factual Density', value: result.detailed_metrics.factual_density * 100 },
    { metric: 'Contradictions', value: result.detailed_metrics.contradiction_score * 100 },
  ] : [];

  const trendData = result ? [
    { name: 'Overall', risk: result.hallucination_probability * 100 },
    { name: 'ML Model', risk: result.detailed_metrics.ml_probability * 100 },
    { name: 'Language', risk: result.detailed_metrics.confidence_issues * 100 },
    { name: 'Facts', risk: (1 - result.detailed_metrics.factual_density) * 100 },
    { name: 'Logic', risk: result.detailed_metrics.contradiction_score * 100 },
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
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <Card className={`border-border ${getRiskLevel(result.hallucination_probability).bg}`}>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Risk Level</p>
                          <p className={`text-2xl font-bold ${getRiskLevel(result.hallucination_probability).color}`}>
                            {(result.hallucination_probability * 100).toFixed(1)}%
                          </p>
                        </div>
                        {result.is_safe ? (
                          <CheckCircle className="h-8 w-8 text-green-400" />
                        ) : (
                          <AlertTriangle className="h-8 w-8 text-red-400" />
                        )}
                      </div>
                      <div className="mt-4">
                        <Badge 
                          variant={result.is_safe ? "secondary" : "destructive"}
                          className="text-xs"
                        >
                          {getRiskLevel(result.hallucination_probability).label}
                        </Badge>
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
                          <p className="text-sm font-medium text-muted-foreground">Processing Time</p>
                          <p className="text-2xl font-bold text-green-400">
                            {result.processing_time ? `${(Date.now() - result.processing_time)}ms` : 'N/A'}
                          </p>
                        </div>
                        <Clock className="h-8 w-8 text-green-400" />
                      </div>
                      <div className="mt-4">
                        <Badge variant="secondary" className="text-xs">
                          {result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : 'Unknown'}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </div>

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
                          />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: '#1F2937', 
                              border: '1px solid #374151',
                              borderRadius: '8px'
                            }}
                          />
                          <Area 
                            type="monotone" 
                            dataKey="risk" 
                            stroke="#ef4444" 
                            fill="#ef4444"
                            fillOpacity={0.3}
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
                    <CardTitle>Detailed Metrics</CardTitle>
                    <CardDescription>
                      Comprehensive analysis breakdown
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { 
                          key: 'ml_probability', 
                          label: 'ML Model Score', 
                          value: result.detailed_metrics.ml_probability,
                          description: 'Machine learning model confidence'
                        },
                        { 
                          key: 'confidence_issues', 
                          label: 'Language Confidence', 
                          value: result.detailed_metrics.confidence_issues,
                          description: 'Overconfident or uncertain language patterns'
                        },
                        { 
                          key: 'factual_density', 
                          label: 'Factual Content', 
                          value: result.detailed_metrics.factual_density,
                          description: 'Density of factual information'
                        },
                        { 
                          key: 'contradiction_score', 
                          label: 'Logical Consistency', 
                          value: result.detailed_metrics.contradiction_score,
                          description: 'Internal contradictions detected'
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
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
                        <CardTitle className="flex items-center space-x-2 text-blue-400">
                          <CheckCircle className="h-5 w-5" />
                          <span>Recommendations</span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-2">
                          {result.recommendations.map((rec, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
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
