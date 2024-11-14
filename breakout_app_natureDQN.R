# app.R
library(shiny)
library(grid)
library(reticulate)

# Initialize Python environment
use_python("~/project/myenv/Scripts/python.exe")

# Python code for game mechanics and DRL agent
py_run_string("
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Set hyperparameters for deep q-learning
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.batch_size = 64
        self.min_replay_size = 128
        
        # Create policy network (for action selection and updates)
        self.policy_model = self._build_model()
        # Create target network (for stable Q-value targets)
        self.target_model = self._build_model()
        # Initialise target network with same weights as policy network
        self.target_model.set_weights(self.policy_model.get_weights())
        
        self.update_target_counter = 0
        self.episodes_trained = 0
        # Update the target network every 10 episodes
        self.update_target_frequency = 10
              
    def _build_model(self):
        # Set up the neural network - same for both policy and target networks
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0  # Add gradient clipping
        )
        
        model.compile(loss='huber', optimizer=optimizer)
        return model
              
    def update_target_model(self):
        # Copy weights from policy network to target network
        self.target_model.set_weights(self.policy_model.get_weights())
              
    # Memory of previous states for the replay          
    def remember(self, state, action, reward, next_state, done):
        state = np.array(state).reshape((1, -1))
        next_state = np.array(next_state).reshape((1, -1))
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        # Action selection uses policy network only
        state = np.array(state).reshape((1, -1))
        # If rand() is less than epsilon, take a random action
        if training and np.random.rand() <= self.epsilon:
            return int(np.random.randint(0, self.action_size))
        # Otherwise use policy network to select the best action
        act_values = self.policy_model.predict(state, verbose=0)
        return int(np.argmax(act_values[0]))
        
    def replay(self):
        # Only start training once we've stored sufficient examples of game state
        if len(self.memory) < self.min_replay_size:
            return {'trained': False, 'loss': 0.0}
            
        if not hasattr(self, 'last_losses'):
            self.last_losses = []
            
        try:
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.vstack([x[0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch])
            next_states = np.vstack([x[3] for x in minibatch])
            dones = np.array([x[4] for x in minibatch])
    
            # Use target network for next state Q-values
            target_q_values = self.target_model.predict(next_states, verbose=0)
            max_target_q = np.max(target_q_values, axis=1)
            targets = rewards + (1 - dones) * self.gamma * max_target_q
            
            # Use policy network for current state Q-values and updates
            current_q = self.policy_model.predict(states, verbose=0)
            for i, action in enumerate(actions):
                current_q[i][action] = targets[i]
    
            # Train policy network
            history = self.policy_model.fit(states, current_q, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.last_losses.append(loss)
            if len(self.last_losses) > 100:
                self.last_losses = self.last_losses[-100:]
                    
            return {'trained': True, 'loss': loss}
                
        except Exception as e:
            print('Error in replay:', str(e))
            return {'trained': False, 'loss': 0.0}
            
    def save_agent(self, filepath):
        # Save both networks
        self.policy_model.save(filepath + '_policy.keras', include_optimizer=True)
        self.target_model.save(filepath + '_target.keras', include_optimizer=True)
        
        import json
        params = {
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate
        }
        with open(filepath + '_params.json', 'w') as f:
            json.dump(params, f)
    
    @classmethod
    def load_agent(cls, filepath):
        agent = cls(state_size=7, action_size=3)
        
        # Load both networks
        agent.policy_model = tf.keras.models.load_model(filepath + '_policy.keras', compile=True)
        agent.target_model = tf.keras.models.load_model(filepath + '_target.keras', compile=True)
        
        import json
        with open(filepath + '_params.json', 'r') as f:
            params = json.load(f)
        
        # Restore parameters
        agent.epsilon = params['epsilon']
        agent.epsilon_min = params['epsilon_min']
        agent.epsilon_decay = params['epsilon_decay']
        agent.gamma = params['gamma']
        agent.learning_rate = params['learning_rate']
        
        return agent

class GameState:
    def __init__(self):
        # Initialise game
        self.current_level = 1
        self.cumulative_score = 0
        self.initial_block_count = 8 * 14
        self.blocks_cleared_threshold = False
        self.steps_since_last_hit = 0      # Track steps without hitting blocks
        self.hits_this_volley = 0          # Track blocks hit since last paddle contact
        self.score_this_volley = 0         # Track total score since last paddle contact
        self.max_steps_without_hit = 50   # Maximum steps allowed without hitting a block
        self.reset()
        
    def are_all_bricks_cleared(self):
        return np.sum(self.bricks) == 0
    
    def reset(self, keep_score=False, next_level=False):
        # Store current score
        current_score = self.score if keep_score else 0
        current_lives = self.lives if next_level else 3  # Keep lives only when advancing levels
        
        # Handle level progression
        if next_level:
            self.current_level += 1
        else:
            self.current_level = 1
        
        self.paddle_pos = 0.5 # Start the paddle in the middle of the game area
        self.blocks_cleared_threshold = False
        
        # Set paddle width based on level (paddle is half the width in the 2nd level)
        self.paddle_width = 0.1 if self.current_level == 1 else 0.05
        
        # Base ball speed
        self.base_speed = 0.01
        self.current_speed = self.base_speed # Can increase or decrease from base
        
        # Ball setup - random positioning
        self.ball_x = np.random.uniform(0.3, 0.7)
        self.ball_y = 0.3
        angle = np.random.uniform(-30, 30)
        angle_rad = np.deg2rad(angle)
        self.ball_dx = self.current_speed * np.sin(angle_rad)
        self.ball_dy = -self.current_speed * np.cos(angle_rad)
        
        # Create brick layout
        self.bricks = np.ones((8, 14))
        
        # Create block colors and points matrix
        self.brick_colors = np.zeros((8, 14), dtype='U6')
        self.brick_points = np.zeros((8, 14), dtype=int)
        
        # Set colors and points by row
        for row in range(8):
            if row > 5:  # Bottom two rows (rows 6-7)
                self.brick_colors[row,:] = 'green'
                self.brick_points[row,:] = 1
            elif row > 3:  # Next two rows (rows 4-5)
                self.brick_colors[row,:] = 'yellow'
                self.brick_points[row,:] = 3
            elif row > 1:  # Next two rows (rows 2-3)
                self.brick_colors[row,:] = 'orange'
                self.brick_points[row,:] = 5
            else:  # Top two rows (rows 0-1)
                self.brick_colors[row,:] = 'red'
                self.brick_points[row,:] = 7
                
        # Restore score
        self.score = current_score
        self.previous_score = current_score
        
        self.lives = current_lives
        self.game_over = False
        self.paddle_speed = 0.08
        self.steps_without_hit = 0
        self.hits_this_volley = 0
        self.score_this_volley = 0
        self.total_steps = 0
        
        return self.get_state()
        
    def check_speed_increase(self):
        # Increase the ball speed after half the blocks are cleared
        if not self.blocks_cleared_threshold:
            remaining_blocks = np.sum(self.bricks)
            if remaining_blocks <= self.initial_block_count * 0.5:
                self.current_speed *= 1.5
                self.blocks_cleared_threshold = True
                # Update ball velocity while maintaining direction
                current_angle = np.arctan2(self.ball_dx, -self.ball_dy)
                self.ball_dx = self.current_speed * np.sin(current_angle)
                self.ball_dy = -self.current_speed * np.cos(current_angle)
    
    def get_state_for_ai(self):
        # Create normalised state vector for AI
        return np.array([[
            self.paddle_pos,          # Paddle position
            self.ball_x,              # Ball X position
            self.ball_y,              # Ball Y position
            self.ball_dx,             # Ball X velocity
            self.ball_dy,             # Ball Y velocity
            np.mean(self.bricks),     # Proportion of remaining bricks
            abs(self.ball_x - self.paddle_pos)  # Distance from paddle to ball
        ]])

    def calculate_reward(self):
        reward = 0
        
        # Substantial reward for breaking bricks
        score_delta = self.score - self.previous_score
        if score_delta > 0:
            #reward += score_delta  # Increased reward for brick breaking
            
            # Reward for hitting bricks - more reward if multiple hits in same volley
            if self.hits_this_volley > 1:
                volley_bonus = self.score_this_volley # Increasing reward for more hits
                reward += volley_bonus
        
        # Reward for keeping paddle close to ball
        if self.ball_y > 0:
            # Better positioning gets better reward
            dist_to_ball = abs(self.ball_x - self.paddle_pos)
            if dist_to_ball < 0.2:  # Within paddle reach
                reward += 1.0 * (1 - dist_to_ball/0.2)  # Max 1.0 reward for perfect position
        
        # Significant penalty for missing ball
        if self.ball_y < 0.05:
            reward -= 5.0
        
        # Game over penalty considers performance
        if self.game_over:
            # Less penalty if more bricks broken
            bricks_remaining = np.sum(self.bricks)
            reward -= 5.0 * bricks_remaining / (8 * 14)
            
        # Level completion rewards
        if self.are_all_bricks_cleared():
            if self.current_level == 1:
                reward += 100.0  # First level completion
            elif self.current_level == 2:
                reward += 200.0  # Second level completion (final)
        
        self.previous_score = self.score
        return reward

    def step_ai(self, action): 
        # Increment steps since last hit
        self.steps_since_last_hit += 1
        
        # Check for speed increase
        self.check_speed_increase()
        
        # Convert action (0, 1, 2) to movement (-1, 0, 1)
        movement = (action - 1) * self.paddle_speed
        
        # Adjust clipping to account for paddle width (allowing paddle to reach edges)
        self.paddle_pos = np.clip(self.paddle_pos + movement, 
                                 self.paddle_width/2,      # Left boundary
                                 1 - self.paddle_width/2)  # Right boundary
        
        # Update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x >= 1:
            self.ball_dx *= -1
        if self.ball_y >= 1:
            self.ball_dy *= -1
            
        # Ball collision with paddle
        paddle_height = 0.02  # Visual height of paddle
        paddle_y = 0.15  # Paddle y position
        
        # Calculate ball position in next frame to prevent tunneling
        next_ball_x = self.ball_x + self.ball_dx
        next_ball_y = self.ball_y + self.ball_dy
        
        # Check if ball will pass through paddle this frame
        if ((self.ball_y >= paddle_y + paddle_height and next_ball_y <= paddle_y + paddle_height) or
            (self.ball_y >= paddle_y and next_ball_y <= paddle_y + paddle_height)):
            
            # Check horizontal collision using current paddle width
            if abs(self.ball_x - self.paddle_pos) < self.paddle_width/2:
                self.ball_y = paddle_y + paddle_height  # Place ball at paddle surface
                self.ball_dy *= -1  # Reverse vertical direction
                
                # Reset volley tracking on paddle hit
                self.hits_this_volley = 0
                self.score_this_volley = 0
                
                # Add paddle momentum
                paddle_momentum = movement * self.paddle_speed
                self.ball_dx += paddle_momentum
                # Ensure ball speed doesn't get too extreme
                self.ball_dx = np.clip(self.ball_dx, -0.05, 0.05)
        
        # Ball collision with bricks
        brick_height = 0.05  # Visual height of each brick
        brick_width = 1.0 / self.bricks.shape[1]  # Width based on 14 columns
        ball_radius = 0.007  # Match the visual ball radius
        
        # Calculate ball's next position
        next_ball_x = self.ball_x + self.ball_dx
        next_ball_y = self.ball_y + self.ball_dy
        
        # Calculate potential brick collisions for current and next position
        # We need to check both to prevent tunneling
        potential_rows = []
        potential_cols = []
        
        # Current position brick coordinates
        curr_row = int((0.9 - self.ball_y) / brick_height)
        curr_col = int(self.ball_x / brick_width)
        
        # Next position brick coordinates
        next_row = int((0.9 - next_ball_y) / brick_height)
        next_col = int(next_ball_x / brick_width)
        
        # Add all potential collision bricks
        for row in [curr_row, next_row]:
            for col in [curr_col, next_col]:
                if (row >= 0 and row < self.bricks.shape[0] and 
                    col >= 0 and col < self.bricks.shape[1]):
                    potential_rows.append(row)
                    potential_cols.append(col)
        
        # Check each potential brick for collision
        collision_occurred = False
        for row, col in zip(potential_rows, potential_cols):
            if (row >= 0 and row < self.bricks.shape[0] and 
                col >= 0 and col < self.bricks.shape[1] and 
                self.bricks[row, col] == 1):
                
                # Calculate brick boundaries
                brick_left = col * brick_width
                brick_right = (brick_left + brick_width)
                brick_top = 0.9 - row * brick_height
                brick_bottom = brick_top - brick_height
                
                # Check for ball intersection with brick (including radius)
                ball_left = self.ball_x - ball_radius
                ball_right = self.ball_x + ball_radius
                ball_top = self.ball_y + ball_radius
                ball_bottom = self.ball_y - ball_radius
                
                # Detailed collision detection
                horizontal_overlap = (ball_right >= brick_left and ball_left <= brick_right)
                vertical_overlap = (ball_top >= brick_bottom and ball_bottom <= brick_top)
                
                if horizontal_overlap and vertical_overlap:
                    # Calculate penetration depths
                    penetration_left = ball_right - brick_left
                    penetration_right = brick_right - ball_left
                    penetration_top = ball_top - brick_bottom
                    penetration_bottom = brick_top - ball_bottom
                    
                    # Find smallest penetration to determine collision side
                    penetrations = [
                        ('left', penetration_left),
                        ('right', penetration_right),
                        ('top', penetration_top),
                        ('bottom', penetration_bottom)
                    ]
                    collision_side, _ = min(penetrations, key=lambda x: x[1])
                    
                    # Handle collision based on side
                    if collision_side in ['left', 'right']:
                        self.ball_dx *= -1
                        # Position correction
                        if collision_side == 'left':
                            self.ball_x = brick_left - ball_radius
                            if self.ball_x < 0:
                                self.ball_x = 0
                                self.ball_dx = abs(self.ball_dx)
                        else:
                            self.ball_x = brick_right + ball_radius
                            if self.ball_x > 1:
                                self_ball_x = 1
                                self.ball_dx = -1 * abs(self.ball_dx)
                    else:  # top or bottom
                        self.ball_dy *= -1
                        # Position correction
                        if collision_side == 'top':
                            self.ball_y = brick_bottom - ball_radius
                        else:
                            self.ball_y = brick_top + ball_radius
                    
                    # Remove brick and update score
                    self.bricks[row, col] = 0
                    self.score += self.brick_points[row, col]
                    self.cumulative_score += self.brick_points[row, col]
                    collision_occurred = True
                    
                    # Update hit tracking
                    self.steps_since_last_hit = 0
                    self.hits_this_volley += 1
                    self.score_this_volley += self.brick_points[row, col]
                    
                    break  # Only handle one collision per frame
        
        # After brick collision section:
        if self.are_all_bricks_cleared():
            reward = self.calculate_reward()  # Get reward including level completion bonus
            if self.current_level == 2:
                # Game complete after second level
                self.game_over = True
                self.reset(keep_score=False)  # Complete reset
            else:
                # Advance to next level
                self.reset(keep_score=True, next_level=True)
            return self.get_state_for_ai(), reward, self.game_over
        
        # Check for loss condition
        if self.ball_y < 0:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self.ball_x = np.random.uniform(0.3, 0.7)  # Start more centrally
                self.ball_y = 0.4
                angle = np.random.uniform(-30, 30)  # Random angle for variety
                angle_rad = np.deg2rad(angle)
                base_speed = 0.01  # Fixed speed
                self.ball_dx = base_speed * np.sin(angle_rad)
                self.ball_dy = -base_speed * np.cos(angle_rad)
        
        # Calculate regular reward and get next state
        reward = self.calculate_reward()
        next_state = self.get_state_for_ai()
    
        return next_state, reward, self.game_over
    
    def get_state(self):
        return {
            'paddle_pos': float(self.paddle_pos),
            'ball_x': float(self.ball_x),
            'ball_y': float(self.ball_y),
            'bricks': self.bricks.tolist(),
            'score': int(self.score),
            'lives': int(self.lives),
            'game_over': bool(self.game_over),
            'level': int(self.current_level)
        }

# Initialise game and AI agent
game = GameState()
state_size = 7  # Size of our state vector
action_size = 3  # Left, Stay, Right
agent = DQNAgent(state_size, action_size)
")

# Create game state in Python
game <- py$game

# UI definition
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
            #gameArea { position: relative; }
            #controls { margin-top: 20px; }
            .recalculating { opacity: 1.0 !important; }
            .metrics-box {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        "))
  ),
  
  titlePanel("Breakout: Deep reinforcement learning"),
  
  fluidRow(
    column(9,
           plotOutput("gameArea", height="700px")
    ),
    column(3,
           div(id="controls",
               actionButton("trainButton", "Start training"),
               actionButton("stopButton", "Stop training"),
               actionButton("resetButton", "Reset agent"),
               actionButton("saveAgent", "Save agent"),
               actionButton("loadAgent", "Load agent"),
               actionButton("watchAI", "Watch AI play"),
               actionButton("stopWatching", "Stop watching"),
               actionButton("resetGame", "Reset game"),
               hr(),
               sliderInput("gameSpeed", "Game speed:",
                           min = 50, max = 2000,
                           value = 200, step = 50),
               hr(),
               div(class="metrics-box",
                   h4("Current game stats"),
                   verbatimTextOutput("currentStats")
               ),
               div(class="metrics-box",
                   h4("Training metrics"),
                   verbatimTextOutput("trainingMetrics")
               )
           )
    )
  )
)

# Server logic
server <- function(input, output, session) {
  # Reactive values
  training_active <- reactiveVal(FALSE)
  watching_ai <- reactiveVal(FALSE)
  current_state <- reactiveVal(game$get_state())
  metrics <- reactiveVal(list(
    episodes = 0,
    max_score = 0,
    current_epsilon = 1.0,
    avg_reward = 0,
    training_count = 0,
    last_loss = 0,
    target_updates = 0
  ))
  
  # Create reactive timer that updates when speed changes
  game_timer <- reactive({
    reactiveTimer(input$gameSpeed)
  })
  
  # Training loop with dual network approach
  observe({
    if (training_active()) {
      invalidateLater(5)
      
      tryCatch({
        # Get current state and action
        state <- py$game$get_state_for_ai()
        action <- py$agent$act(state)
        
        # Take action and observe result
        result <- py$game$step_ai(as.integer(action))
        next_state <- result[[1]]
        reward <- result[[2]]
        done <- result[[3]]
        
        # Store experience
        py$agent$remember(state, action, reward, next_state, done)
        
        # Update metrics object
        m <- metrics()
        m$current_epsilon <- py$agent$epsilon
        m$max_score <- max(m$max_score, py$game$score)
        
        if (done) {
          # Train on multiple batches when episode ends
          num_training_steps <- 5  # Reduced from 10 for stability
          total_loss <- 0
          
          for(i in 1:num_training_steps) {
            replay_result <- py$agent$replay()
            if (replay_result$trained) {
              m$training_count <- m$training_count + 1
              total_loss <- total_loss + replay_result$loss
            }
          }
          
          # Average loss over training steps
          m$last_loss <- total_loss / num_training_steps
          
          m$episodes <- m$episodes + 1
          
          # Update target network periodically
          if (m$episodes %% py$agent$update_target_frequency == 0) {
            py$agent$update_target_model()
            m$target_updates <- m$target_updates + 1
          }
          # Training over so reset the game
          py$game$reset()
        }
        
        # Update metrics every step
        metrics(m)
        
      }, error = function(e) {
        curr_error <- conditionMessage(e)
        if (!exists("last_error") || curr_error != last_error) {
          cat("Error in training loop:", curr_error, "\n")
          last_error <<- curr_error
        }
      })
    }
  })
  
  # Separate UI update handler - calls the get_state function at intervals
  observe({
    if (training_active()) {
      game_timer()()  # Call the timer function
      current_state(py$game$get_state())
    }
  })
  
  # Watch AI play (uses policy network only for actions)
  observe({
    if (watching_ai()) {  
      game_timer()()  # Call the timer function
      
      # Get action from trained agent (policy network)
      state <- py$game$get_state_for_ai()
      action <- py$agent$act(state, training=FALSE)  # No exploration during watching
      
      # Take action
      result <- py$game$step_ai(as.integer(action))
      current_state(py$game$get_state())
      
      if (result[[3]]) {  # if game over
        py$game$reset()
      }
    }
  })
  
  # Start training
  observeEvent(input$trainButton, {
    training_active(TRUE)
  })
  
  # Stop training
  observeEvent(input$stopButton, {
    training_active(FALSE)
  })
  
  # Reset agent (initialises both networks and resets the game)
  observeEvent(input$resetButton, {
    py$agent <- py$DQNAgent(py$state_size, py$action_size)
    py$game$reset()
    current_state(py$game$get_state())
    metrics(list(
      episodes = 0,
      max_score = 0,
      current_epsilon = 1.0,
      avg_reward = 0,
      training_count = 0,
      last_loss = 0,
      target_updates = 0
    ))
  })
  
  # Save agent (saves both networks)
  observeEvent(input$saveAgent, {
    py$agent$save_agent("./trained_breakout_agent")
  })
  
  # Load agent (loads both networks)
  observeEvent(input$loadAgent, {
    py$agent <- py$DQNAgent$load_agent("./trained_breakout_agent")
  })
  
  # Watch AI handler
  observeEvent(input$watchAI, {
    watching_ai(TRUE)
  })
  
  # Stop watching handler
  observeEvent(input$stopWatching, {
    watching_ai(FALSE)
  })
  
  # Reset game handler
  observeEvent(input$resetGame, {
    py$game$reset()
    current_state(py$game$get_state())
  })
  
  # Render game
  output$gameArea <- renderPlot({
    state <- current_state()
    
    grid.newpage()
    pushViewport(viewport(xscale=c(0,1), yscale=c(0,1)))
    
    # Draw background
    grid.rect(gp=gpar(fill="black"))
    
    # Get brick colors from Python
    brick_colors <- py$game$brick_colors
    
    # Draw bricks
    bricks_matrix <- matrix(unlist(state$bricks), 
                            nrow=8, ncol=14, byrow=TRUE)
    for(row in 1:nrow(bricks_matrix)) {
      for(col in 1:ncol(bricks_matrix)) {
        if(bricks_matrix[row,col] == 1) {
          grid.rect(x = (col-0.5)/14,
                    y = 0.875 - (row-1)*0.05,  # Changed from 1.0 to 0.9 to create gap at top
                    width = 1/14,
                    height = 0.05,
                    gp = gpar(fill=brick_colors[row,col], 
                              col="black",
                              lwd=0.5),
                    default.units = "native")
        }
      }
    }
    
    # Draw paddle 
    grid.rect(x = state$paddle_pos,
              y = 0.15,
              width = py$game$paddle_width,  # Use paddle width from game state
              height = 0.02,
              gp = gpar(fill="steelblue2"),
              default.units = "native")
    
    # Draw ball
    grid.circle(x = state$ball_x,
                y = state$ball_y,
                r = 0.007,        # Changed from 0.01 to 0.007
                gp = gpar(fill="white"),
                default.units = "native")
  })
  
  # Current game stats
  output$currentStats <- renderText({
    state <- current_state()
    paste0(
      "Current Score: ", state$score, "\n",
      "Lives: ", state$lives, "\n",
      "Ball Position: (", 
      round(state$ball_x, 2), ", ", 
      round(state$ball_y, 2), ")\n",
      "Paddle Position: ", round(state$paddle_pos, 2)
    )
  })
  
  # Training metrics display
  output$trainingMetrics <- renderText({
    m <- metrics()
    paste0(
      "Episodes: ", m$episodes, "\n",
      "Max score: ", m$max_score, "\n",
      "Epsilon: ", round(m$current_epsilon, 4), "\n",
      "Training steps: ", m$training_count, "\n",
      "Target updates: ", m$target_updates, "\n",
      "Last loss: ", round(m$last_loss, 6)
    )
  })

}

# Run the app
shinyApp(ui = ui, server = server)