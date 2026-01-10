# RLM Interactive Visualization

## How to Use

Simply open `rlm-visualization.html` in your web browser. No build step or server required!

```bash
# Option 1: Open directly
open rlm-visualization.html  # macOS
xdg-open rlm-visualization.html  # Linux
start rlm-visualization.html  # Windows

# Option 2: Use a local server
python -m http.server 8000
# Then visit: http://localhost:8000/rlm-visualization.html
```

## Features

### 🎬 12-Step Animated Workflow
Watch the complete RLM execution process unfold:
1. **Query Input** - Initial query enters the system
2. **Root LM Activation** - Main language model starts processing
3. **REPL Environment** - Context storage and function injection
4. **Code Generation** - LM creates decomposition strategy
5. **Spawn Sub-Query 1** - First recursive call
6. **Sub-LM 1 Processing** - Focused sub-problem solving
7. **Spawn Sub-Query 2** - Parallel recursive call
8. **Sub-LM 2 Processing** - Another sub-problem branch
9. **Deep Recursion** - Demonstrates depth=2 capability
10. **Response Propagation** - Results flow back up
11. **Final Synthesis** - Root LM combines all responses
12. **Complete** - Full system visualization

### 🎮 Interactive Controls

- **▶ Play** - Auto-advance through all steps (3s intervals)
- **⏸ Pause** - Pause the animation
- **↻ Reset** - Start over from the beginning
- **→ Next Step** - Manually advance one step at a time

### 🎨 Visual Elements

- **Blue Spheres** - Language Model instances
- **Pink Cubes** - REPL Environments
- **Blue Particles** - Query data flowing (questions)
- **Green Particles** - Response data flowing (answers)
- **Connecting Lines** - Data pathways
- **Pulsing Effects** - Active processing
- **Rotating Camera** - Automatic 3D rotation for better viewing

### 🖱️ Mouse Interaction

- **Hover over nodes** - See tooltips with node information
- **Watch particle flow** - Observe how queries and responses travel through the system

## What You'll Learn

This visualization demonstrates:

1. **Hierarchical Decomposition** - How complex queries split into manageable sub-queries
2. **Recursive Architecture** - LMs calling other LMs at multiple depths
3. **Context Management** - How context is stored separately and accessed selectively
4. **Parallel Processing** - Multiple sub-queries executing simultaneously
5. **Response Aggregation** - How sub-results combine into the final answer
6. **Scalability** - Why this approach works with unlimited context length

## Technical Details

- **Built with**: Three.js (r128)
- **Rendering**: WebGL with shadow mapping
- **Animation**: Custom particle system and tween interpolation
- **No dependencies**: Single HTML file, works offline (CDN for Three.js)

## Color Coding

- 🔵 **Purple (#667eea)** - Language Models
- 🌸 **Pink (#f093fb)** - REPL Environments
- 💠 **Blue (#4facfe)** - Query flow (questions going down)
- 🟢 **Green (#43e97b)** - Response flow (answers coming up)

## Tips for Best Experience

1. **Use a modern browser** - Chrome, Firefox, Safari, Edge (latest versions)
2. **Let it auto-play first** - Get the full narrative experience
3. **Then use manual stepping** - Study each step in detail
4. **Read the step descriptions** - Upper right panel explains what's happening
5. **Check the progress bar** - Know where you are in the sequence

Enjoy exploring how Recursive Language Models work! 🚀
