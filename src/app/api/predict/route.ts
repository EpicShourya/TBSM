import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

interface PredictionRequest {
  text: string;
  context?: string;
}

interface PredictionResponse {
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
}

async function runPythonScript(text: string, context?: string): Promise<PredictionResponse> {
  return new Promise((resolve, reject) => {
    // Path to the Python script
    const scriptPath = path.join(process.cwd(), 'src', 'ml', 'hallucination_detector.py');
    
    // Spawn Python process with UTF-8 encoding
    const pythonProcess = spawn('python', [scriptPath, text], {
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8'
      }
    });
    
    let stdoutData = '';
    let stderrData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          // Clean the output by removing installation messages and warnings
          const cleanOutput = stdoutData
            .split('\n')
            .filter(line => !line.includes('Defaulting to user installation'))
            .filter(line => !line.includes('Requirement already satisfied'))
            .filter(line => !line.includes('[notice]'))
            .filter(line => !line.includes('OK:'))
            .filter(line => !line.includes('WARNING:'))
            .filter(line => !line.includes('ERROR:'))
            .filter(line => line.trim().length > 0)
            .join('\n')
            .trim();
          
          // Find JSON object
          let jsonStr = '';
          const lines = cleanOutput.split('\n');
          let bracesCount = 0;
          let inJson = false;
          
          for (const line of lines) {
            if (line.trim().startsWith('{')) {
              inJson = true;
              bracesCount = 1;
              jsonStr = line;
            } else if (inJson) {
              jsonStr += '\n' + line;
              // Count braces to find end of JSON
              for (const char of line) {
                if (char === '{') bracesCount++;
                if (char === '}') bracesCount--;
              }
              if (bracesCount === 0) break;
            }
          }
          
          if (!jsonStr) {
            jsonStr = cleanOutput;
          }
          
          const result = JSON.parse(jsonStr);
          resolve(result);
        } catch (parseError) {
          console.error('Parse error:', parseError);
          console.error('Raw stdout:', stdoutData);
          console.error('Stderr:', stderrData);
          reject(new Error(`Failed to parse JSON: ${parseError}`));
        }
      } else {
        console.error('Python script failed:', stderrData);
        reject(new Error(`Python script exited with code ${code}: ${stderrData}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });
  });
}

export async function POST(request: NextRequest) {
  try {
    const body: PredictionRequest = await request.json();
    
    if (!body.text || typeof body.text !== 'string') {
      return NextResponse.json(
        { error: 'Text is required and must be a string' },
        { status: 400 }
      );
    }
    
    // Validate text length
    if (body.text.length > 10000) {
      return NextResponse.json(
        { error: 'Text too long (max 10,000 characters)' },
        { status: 400 }
      );
    }
    
    console.log('Processing text:', body.text.substring(0, 100) + '...');
    
    // Run Python script
    const result = await runPythonScript(body.text, body.context);
    
    // Add timestamp
    const response = {
      ...result,
      timestamp: new Date().toISOString(),
      processing_time: Date.now()
    };
    
    return NextResponse.json(response);
    
  } catch (error) {
    console.error('Prediction error:', error);
    
    return NextResponse.json(
      { 
        error: 'Failed to analyze text',
        details: error instanceof Error ? error.message : 'Unknown error',
        hallucination_probability: 0.5,
        confidence_score: 0.5,
        detected_issues: ['Analysis failed - please try again'],
        recommendations: ['Check your input and try again'],
        is_safe: false,
        detailed_metrics: {
          confidence_issues: 0,
          factual_density: 0,
          contradiction_score: 0,
          ml_probability: 0.5
        }
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Hallucination Detection API',
    version: '1.0.0',
    endpoints: {
      'POST /api/predict': 'Analyze text for hallucinations'
    },
    usage: {
      method: 'POST',
      body: {
        text: 'string (required) - Text to analyze',
        context: 'string (optional) - Additional context'
      }
    }
  });
}
